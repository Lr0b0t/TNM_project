function extract_timeseries_by_label(func_dir::String, func_fileName::String, atlas_dir::String, atlas_fileName::String, subject::String)
    # === 1. Construct full file paths for functional and atlas NIfTI files ===
    func_path = joinpath(func_dir, subject, func_fileName)
    println(func_path)  # Optional: for debugging, prints the functional file path
    atlas_path = joinpath(atlas_dir, atlas_fileName)

    # === 2. Load NIfTI images ===
    func_nii  = niread(func_path)   # 4D functional image: (X, Y, Z, T)
    atlas_nii = niread(atlas_path)  # 3D atlas image: (X, Y, Z), with integer labels for regions

    # === 3. Extract raw voxel data from NIfTI objects ===
    vol4d = func_nii.raw            # Functional data array: Float32[X, Y, Z, T]
    atlas = atlas_nii.raw           # Atlas labels 

    dims3 = size(atlas)             # Get spatial dimensions (X, Y, Z)
    T     = size(vol4d, 4)          # Number of time points in functional data
    TR    = time_step(func_nii.header) / 1000  # Convert repetition time from ms to seconds

    # === 4. Reshape functional and atlas data for easier processing ===
    vol2d   = reshape(vol4d, prod(dims3), T)  # Reshape to 2D: (voxels × time)
    atlas1  = vec(atlas)                      # Flatten atlas to 1D: (voxels)

    # === 5. Identify unique non-zero labels in atlas (i.e., brain regions) ===
    regs = sort(unique(atlas1))              # All unique labels
    regs = filter(!=(0), regs)               # Remove background (label 0)

    # === 6. Initialize output matrix for time series data ===
    nreg = length(regs)                      # Number of brain regions (ROIs)
    ts   = zeros(Float64, nreg, T)           # Preallocate array: (regions × time)

    # === 7. For each brain region, average the functional signal across all voxels ===
    for (i, lbl) in enumerate(regs)
        mask_vec = atlas1 .== lbl            # Create a mask for voxels in region `lbl`

        if !any(mask_vec)
            @warn "No voxels found for label $lbl in atlas"  # Warn if region is empty
            continue
        end

        roi_data = vol2d[mask_vec, :]        # Extract time series from selected voxels
        ts[i, :] = mean(roi_data; dims=1)[:] # Compute average time series for the region
    end

    # === 8. Return results ===
    # ts   : matrix of averaged time series (regions × time)
    # regs : list of region labels used
    # TR   : time between functional image acquisitions (in seconds)
    return ts, regs, TR
end



function compute_functional_connectivity(ts::AbstractMatrix{<:Real};
                                         method::Symbol = :pearson,
                                         fisher_z::Bool = false)
""" 
Computes the functional connectivity matrix between brain regions based on their time series.

# Inputs:
- `ts`: A matrix of shape (nROI × T), where each row is the time series of a brain region (ROI),
        and T is the number of time points.
- `method` (optional): A `Symbol` specifying the connectivity measure.
    - `:pearson` (default): Computes the Pearson correlation coefficient between each pair of ROIs.
    - `:covariance`: Computes the covariance matrix instead.
- `fisher_z` (optional): A `Bool` indicating whether to apply Fisher Z-transformation
                         to the Pearson correlations (ignored if method ≠ :pearson).

# Output:
- Returns an `nROI × nROI` symmetric matrix `C` where:
    - `C[i, j]` is the correlation or covariance between ROI `i` and ROI `j`.
    - If `fisher_z = true`, correlations are transformed using Fisher's Z (atanh).

"""
    nroi, _ = size(ts)
    C = zeros(Float64, nroi, nroi)  # Initialize connectivity matrix

    if method == :pearson
        # Compute Pearson correlation between all pairs of ROIs
        # Optionally apply Fisher Z-transform for statistical normalization
        for i in 1:nroi
            for j in i:nroi
                r = cor(ts[i, :], ts[j, :])
                if fisher_z
                    # Clamp values to avoid infinities when applying atanh
                    r_clamped = clamp(r, -0.9999, 0.9999)
                    val = atanh(r_clamped)
                else
                    val = r
                end
                C[i, j] = val
                C[j, i] = val  # Ensure symmetry
            end
        end

    elseif method == :covariance
        # Directly compute covariance matrix between ROIs
        # Note: ts' transposes to T × nROI for cov to work correctly
        C = cov(ts')

    else
        throw(ArgumentError("Unsupported method: $method; choose :pearson or :covariance"))
    end

    return C
end

function estimate_rdcm(ts::Matrix{Float64}, TR::Float64; verbose::Int=1, A::BitMatrix=nothing)
"""

Estimates a Rigid Regression Dynamic Causal Model (rDCM) from regional BOLD time series.

# Inputs:
- `ts`: A matrix of shape (nROI × T), where each row corresponds to the BOLD time series 
        of a brain region (ROI) and each column is a time point.
- `TR`: Repetition time (in seconds) of the fMRI acquisition.
- `verbose` (optional): Integer to control verbosity of output logs. Default is 1.
- `A` (optional): A binary (logical) adjacency matrix indicating which connections should 
                  be estimated. 

# Output:
- `rdcm`: The estimated rigid DCM model object.
- `output`: Results of the model inversion, including estimated parameters and diagnostics.
"""
    # 1) Ensure self-connections are present in A (necessary for model stability)

    nroi, scans = size(ts)
    for i in 1:nroi
        A[i,i] = true
    end

    # 2) no driving inputs
    C = falses(nroi, 0)

    # 3) package the BOLD data: expects scans×regions
    Y    = RegressionDynamicCausalModeling.BoldY(ts', TR, nothing)  # :contentReference[oaicite:1]{index=1}

    # 4) no confounds
    r_dt = 16
    u_dt = TR / r_dt
    X_numOfSteps =  round(Int64, scans * TR / u_dt)

    Conf = RegressionDynamicCausalModeling.Confound(zeros(X_numOfSteps, 0), String[])

    # 5) zero initial parameters for A and C
    Ep   = RegressionDynamicCausalModeling.TrueParamLinear(
               zeros(nroi, nroi),      # A‐parameters
               zeros(nroi, 0)           # C‐parameters
           )

    # 6) build a LinearDCM and convert to rigid rDCM
    dcm  = RegressionDynamicCausalModeling.LinearDCM(A, C, scans, nroi, nothing, Y, Ep, Conf)
    rdcm = RegressionDynamicCausalModeling.RigidRdcm(dcm)             # :contentReference[oaicite:2]{index=2}

    # 7) set inversion options and invert
    invpars = RegressionDynamicCausalModeling.RigidInversionParams()   # default maxIter, tol
    opt     = RegressionDynamicCausalModeling.Options(invpars; synthetic=false, verbose=verbose)
    output  = RegressionDynamicCausalModeling.invert(rdcm, opt)        # :contentReference[oaicite:3]{index=3}

    return rdcm, output
end


function extract_allSubjectIDs(
    func_dir::String
)
"""
Scans a functional data directory and returns a list of subject IDs.

# Input:
- `func_dir`: Path to the directory containing one subfolder per subject.

# Output:
- A vector of subject folder names (IDs) extracted from the directory structure.
"""
    # Get subject folders and IDs
    subject_folders = filter(isdir, readdir(func_dir, join=true))
    subject_ids = [basename(folder) for folder in subject_folders]
    return subject_ids
end 


function extract_timeseries_for_all_subjects(
    func_dir::String,
    results_dir::String,
    func_fileName::String,
    atlas_dir::String,
    atlas_fileName::String;
    verbose::Bool = true
)
""" 

Extracts BOLD time series for all subjects in a dataset using a provided brain atlas,
and saves the result for each subject to a `.mat` file.

# Inputs:
- `func_dir`: Directory containing subject subfolders with functional NIfTI files.
- `results_dir`: Directory where output `.mat` files will be saved.
- `func_fileName`: Name of the 4D functional NIfTI file located inside each subject folder.
- `atlas_dir`: Directory containing the atlas NIfTI file.
- `atlas_fileName`: Name of the 3D brain atlas NIfTI file.
- `verbose` (optional): If `true`, prints progress updates. Default is `true`.

# Output:
- Saves a `time_series.mat` file for each subject in `results_dir/<subject>/`.
  Each `.mat` file contains:
    - `"ts"`: ROI × time matrix of BOLD signals averaged within atlas regions.
    - `"labels"`: List of atlas region labels.
    - `"TR"`: Repetition time (seconds).
"""
    # Get subject folders and IDs
    subject_folders = filter(isdir, readdir(func_dir, join=true))
    subject_ids = [basename(folder) for folder in subject_folders]

    for subject in subject_ids
        if verbose
            @printf("Extracting time series for subject: %s\n", subject)
        end

        # Call your custom function to extract time series (make sure this function is defined elsewhere)
        ts, labels, TR = extract_timeseries_by_label(
            func_dir, func_fileName,
            atlas_dir, atlas_fileName,
            subject
        )

        # Save path for the .mat file
        save_path = joinpath(results_dir, subject, "time_series.mat")
        mkpath(dirname(save_path))  # Create directories if needed

        # Save the results
        MAT.matwrite(save_path, Dict(
            "ts" => ts,
            "labels" => labels,
            "TR" => TR
        ))
    end

    if verbose
        println("Finished processing all subjects.")
    end
end

# === Function to compute and save functional connectivity ===
function compute_func_connectivity_for_all_subjects(results_dir::String; verbose::Bool=true)
"""
Computes functional connectivity matrices for all subjects based on previously extracted
regional BOLD time series and saves the results.

# Input:
- `results_dir`: Directory containing subfolders for each subject with a `time_series.mat` file.
- `verbose` (optional): If `true`, prints progress messages. Default is `true`.

# Output:
- For each subject, saves a `func_connectivity.mat` file containing:
    - `"fc_mat"`: Pearson correlation matrix.
    - `"fc_z"`: Fisher Z-transformed correlation matrix.
    - `"cov_mat"`: Covariance matrix of the time series.
"""


    subject_folders = filter(isdir, readdir(results_dir, join=true))
    subject_ids = [basename(folder) for folder in subject_folders]

    for subject in subject_ids
        if verbose
            @printf("Computing functional connectivity for subject: %s\n", subject)
        end

        # Load extracted timeseries
        ts_data = MAT.matread(joinpath(results_dir, subject, "time_series.mat"))
        ts = ts_data["ts"]

        fc_mat = compute_functional_connectivity(ts)
        fc_z = compute_functional_connectivity(ts; fisher_z=true)
        cov_mat = compute_functional_connectivity(ts; method=:covariance)

        save_path = joinpath(results_dir, subject, "func_connectivity.mat")
        mkpath(dirname(save_path))

        MAT.matwrite(save_path, Dict(
            "fc_mat" => fc_mat,
            "fc_z" => fc_z,
            "cov_mat" => cov_mat
        ))
    end

    if verbose
        println("Finished computing connectivity for all subjects.")
    end
end



function estimate_rdcm_for_all_subjects(
    results_dir::String,
    atlas_filename::String,
    bna_matrix_filename::String;
    verbose::Bool = true
)


"""

Estimates Rigid Regression Dynamic Causal Models (rDCMs) for all subjects using previously 
extracted BOLD time series and a provided structural connectivity matrix (adjacency matrix).

# Inputs:
- `results_dir`: Directory containing subject folders with `time_series.mat` files.
- `atlas_filename`: (Currently unused) Placeholder for possible future atlas metadata.
- `bna_matrix_filename`: CSV file containing the structural adjacency matrix (e.g., from BNA atlas).
- `verbose` (optional): If `true`, prints progress updates.

# Output:
- For each subject, saves a file `rdcm_connectivity.mat` in their respective folder, containing:
    - `"output_m_all"`: The estimated model parameters across iterations (from rDCM output).
"""
    # === 1. Load structural adjacency matrix ===
    A_array = readdlm(bna_matrix_filename, ',', Int)

    A_matrix = BitMatrix(A_array .== 1)

    # === 2. Get list of subjects ===

    subject_folders = filter(isdir, readdir(results_dir, join=true))
    subject_ids = [basename(folder) for folder in subject_folders]


    # === 3. Estimate rDCM for each subject ===
    for subject in subject_ids
        if verbose
            @printf("\nProcessing subject: %s\n", subject)
        end
        # === 4. Load BOLD data and metadata ===
        ts_path = joinpath(results_dir, subject, "time_series.mat")
        if !isfile(ts_path)
            @warn "Skipping subject $subject: time series file not found."
            continue
        end

        data = MAT.matread(ts_path)
        ts = data["ts"]
        labels = data["labels"]
        TR = data["TR"]

        # === 5. Estimate the rDCM model ===
        rdcm, output = estimate_rdcm(ts, TR, verbose=  Int64(verbose::Bool) , A=A_matrix)

        # === 6. Save rDCM output ===
        save_path = joinpath(results_dir, subject, "rdcm_connectivity.mat")
        mkpath(dirname(save_path))

        MAT.matwrite(save_path, Dict(
            "output_m_all" => output.m_all
        ))
    end

    println("\nFinished estimating rDCM for all subjects.")
end

function load_connectivity_data(
    results_dir::String;
    load_rdcm::Bool = true,
    load_functional::Bool = true,
    verbose::Bool = true
)
""" 
Loads precomputed connectivity data (functional and/or rDCM) for all subjects from the results directory.

# Inputs:
- `results_dir`: Directory containing per-subject folders with connectivity `.mat` files.
- `load_rdcm`: If `true`, attempts to load rDCM results (`rdcm_connectivity.mat`).
- `load_functional`: If `true`, attempts to load functional connectivity (`func_connectivity.mat`).
- `verbose`: If `true`, prints progress and warnings to the console.

# Output:
- Returns a dictionary `all_data` where:
    - Keys are subject IDs (strings),
    - Values are dictionaries with keys like `"rdcm"`, `"fc_mat"`, `"fc_z"`, and `"cov_mat"` depending on what was loaded.
"""

    subject_folders = filter(isdir, readdir(results_dir, join=true))
    subject_ids = [basename(folder) for folder in subject_folders]

    all_data = Dict{String, Dict{String, Any}}()

    for subject in subject_ids
        subj_data = Dict{String, Any}()

        if load_rdcm
            rdcm_path = joinpath(results_dir, subject, "rdcm_connectivity.mat")

            if isfile(rdcm_path)
                if verbose
                    @printf("Loading rDCM output for subject: %s\n", subject)
                end
                data = MAT.matread(rdcm_path)
                if haskey(data, "output_m_all")
                    subj_data["rdcm"] = data["output_m_all"]
                else
                    @warn "Missing 'output_m_all' in rDCM file for subject $subject"
                end
            else
                @warn "rDCM file not found for subject $subject"
            end
        end

        if load_functional
            func_path = joinpath(results_dir, subject, "func_connectivity.mat")

            if isfile(func_path)
                if verbose
                    @printf("Loading functional connectivity for subject: %s\n", subject)
                end
                data = MAT.matread(func_path)
                for k in ["fc_mat", "fc_z", "cov_mat"]
                    if haskey(data, k)
                        subj_data[k] = data[k]
                    else
                        @warn "Missing $k in functional connectivity file for subject $subject"
                    end
                end
            else
                @warn "Functional connectivity file not found for subject $subject"
            end
        end

        if !isempty(subj_data)
            all_data[subject] = subj_data
        end
    end

    println("\nFinished loading connectivity data.")
    return all_data
end


function mask_and_flatten_connectivity_data(connectivity_data::Dict{String, Dict{String, Any}},
                                            bna_matrix_filename::String;
                                            mask::Bool = true)
"""
Applies a binary network mask to connectivity matrices ( from BNA atlas),
and flattens the results into 1D vectors for further analysis ( machine learning or statistics).

# Inputs:
- `connectivity_data`: A dictionary where each key is a subject ID, and each value is a dictionary
  with connectivity matrices (e.g., `"rdcm"`, `"fc_mat"`, `"fc_z"`, `"cov_mat"`).
- `bna_matrix_filename`: Path to a CSV file containing a binary structural mask (246x246).
- `mask` (optional): If `true` (default), apply the mask to extract only relevant connections.
  If `false`, flatten the full matrix regardless of structure.

# Output:
- Returns a dictionary with the same subject keys, where each subject maps to a dictionary:
    - Keys are connectivity types (e.g., `"rdcm"`, `"fc_z"`),
    - Values are 1D vectors (masked or fully flattened).
"""
    # === 1. Load the BNA mask as a BitMatrix ===
    A_array = readdlm(bna_matrix_filename, ',', Int)
    A_matrix = BitMatrix(A_array .== 1)

    # Keys corresponding to symmetric matrices (only upper triangle used)
    symmetric_keys = Set(["fc_mat", "fc_z", "cov_mat"])

    # === 2. Initialize output dictionary ===
    masked_data = Dict{String, Dict{String, Vector{Float64}}}()

    # === 3. Loop through all subjects ===
    for (subject, subj_data) in connectivity_data
        masked_subj = Dict{String, Vector{Float64}}()

        # === 4. Process each type of connectivity matrix ===
        for key in ["rdcm", "fc_mat", "fc_z", "cov_mat"]
            if haskey(subj_data, key)
                mat = Array{Float64}(subj_data[key])  # Ensure matrix is of correct type

                # === 5. Generate mask indices ===
                if key in symmetric_keys
                    # Use only upper triangle for symmetric matrices to avoid duplication
                    upper_mask = triu(A_matrix)
                    mask_idx = findall(upper_mask .== 1)
                else
                    # For asymmetric matrices, use full mask
                    mask_idx = findall(A_matrix .== 1)
                end

                # === 6. Apply mask or flatten ===
                flat_vec = mask ? mat[mask_idx] : vec(mat)

                # === 7. Store the flattened result ===
                masked_subj[key] = flat_vec
            end
        end

        # Add to result if any data was found
        masked_data[subject] = masked_subj
    end

    return masked_data
end


function extract_allSubjectIDs(func_dir::String)
    """

    Scans a directory containing subject folders and returns a list of subject IDs (folder names).

    # Input:
    - `func_dir`: Path to the directory where each subject has its own subfolder.

    # Output:
    - A vector of subject IDs (folder names).
    """
    # List only subdirectories in the given directory
    subject_folders = filter(isdir, readdir(func_dir, join=true))
    # Extract the folder names (subject IDs)
    subject_ids = [basename(folder) for folder in subject_folders]
    return subject_ids
end


function select_subjects_by_group(
    data::Matrix{Float64},
    group_array::Vector{Int},
    group_counts::Vector{Int},
    preselected::Bool = false,
    preselected_indices::Vector{Int} = Int[]
)

    """
    Selects a subset of subjects from a dataset based on group labels, either by random sampling
    or using a predefined set of indices.

    # Inputs:
    - `data`: A matrix where rows are subjects and columns are features.
    - `group_array`: A vector indicating the group assignment of each subject (e.g., 1, 2, ...).
    - `group_counts`: A vector indicating how many subjects to select from each group.
    - `preselected` (optional): If `true`, use `preselected_indices` instead of random sampling.
    - `preselected_indices` (optional): Indices of preselected subjects to use when `preselected = true`.

    # Output:
    - `selected_data`: Subset of `data` corresponding to selected subjects.
    - `selected_groups`: Group labels for the selected subjects.
    - `selected_idxs`: Row indices in the original data matrix for the selected subjects.
    """


    data_dim = size(data, 2)  # Number of features (columns)

    if preselected
        # === Use preselected indices directly ===
        total_selected = length(preselected_indices)
        selected_data = data[preselected_indices, :]
        selected_groups = group_array[preselected_indices]
        selected_idxs = preselected_indices

    else
        # === Randomly sample subjects per group ===
        total_selected = sum(group_counts)
        selected_data = Array{Float64}(undef, total_selected, data_dim)
        selected_groups = Vector{Int}(undef, total_selected)
        selected_idxs = Vector{Int}(undef, total_selected)

        insert_pos = 1  # Tracks where to insert the next block of selected samples

        for group_id in 1:length(group_counts)
            group_indices = findall(x -> x == group_id, group_array)
            n_available = length(group_indices)
            n_to_select = group_counts[group_id]

            if n_to_select > n_available
                error("Requested $n_to_select subjects from group $group_id, but only $n_available available.")
            end

            selected = randperm(n_available)[1:n_to_select]
            selected_inds = group_indices[selected]

            # Store selected rows and metadata
            selected_data[insert_pos:insert_pos+n_to_select-1, :] = data[selected_inds, :]
            selected_groups[insert_pos:insert_pos+n_to_select-1] = group_array[selected_inds]
            selected_idxs[insert_pos:insert_pos+n_to_select-1] = selected_inds

            insert_pos += n_to_select
        end
    end

    return selected_data, selected_groups, selected_idxs
end


function get_connectivity_matrix(masked_connectivity_vectors::Dict{String, Dict{String, Vector{Float64}}}, 
                                  subjectIDs::Vector{String}, 
                                  feature_key::String = "rdcm")
    """
    Constructs a subject-by-feature matrix from flattened connectivity vectors 
    (e.g., rDCM or functional connectivity) stored in a nested dictionary.

    # Inputs:
    - `masked_connectivity_vectors`: A dictionary where each key is a subject ID, 
    and each value is another dictionary with connectivity feature vectors 
    (e.g., "rdcm", "fc_mat").
    - `subjectIDs`: Ordered list of subject IDs to include in the matrix.
    - `feature_key` (optional): Which type of connectivity data to extract 
    (default = "rdcm").

    # Output:
    - A matrix of shape (n_subjects × n_features), where each row corresponds to one subject's vector.
    """

    data = []

    # === 1. Loop over subject IDs and extract connectivity vectors ===
    for id in subjectIDs
        if haskey(masked_connectivity_vectors, id) && haskey(masked_connectivity_vectors[id], feature_key)
            push!(data, masked_connectivity_vectors[id][feature_key])
        else
            error("Missing connectivity data for subject ID: $id or feature key: $feature_key")
        end
    end

    # === 2. Convert list of vectors into a subject × feature matrix ===
    # Transpose each row vector, then vertically concatenate into matrix
    return reduce(vcat, [x' for x in data])
end
