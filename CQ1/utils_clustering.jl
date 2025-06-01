"""
    train_vae_model(data::AbstractArray{Float32}, latent_dim::Int, num_epochs::Int)

Trains a Variational Autoencoder (VAE) using the AutoEncoderToolkit.jl on subject-wise 
connectivity data.

# Inputs:
- `data`: A Float32 array (subjects × features), typically derived from connectivity matrices.
- `latent_dim`: The size of the latent representation (number of dimensions).
- `num_epochs`: Number of training epochs.

# Output:
- Returns the trained `vae` model and a named tuple of training/validation metrics:
  (`train_loss`, `val_loss`, `train_mse`, `val_mse`, `train_entropy`, `val_entropy`)

Note: GPU acceleration is used if available. If not, the model runs on CPU.
"""
function train_vae_model(data::AbstractArray{Float32}, latent_dim::Int, num_epochs::Int)

    # === 1. Prepare data ===
    # Transpose so features are rows and samples are columns (Flux convention)
    connectivity_data = transpose(data)

    # Split into training and validation sets
    train_data = connectivity_data[:, 1:80]  
    val_data = connectivity_data[:, 81:end]  
    input_dim = size(train_data, 1)

    # === 2. Network architecture ===
    # Define encoder: input -> dense layers -> mean/logvar
    dense_layers = Flux.Chain(
        Dense(input_dim, 400, relu),
        Dense(400, 256)
    )

    encoder = AutoEncoderToolkit.JointGaussianLogEncoder(
        dense_layers,
        Dense(256, latent_dim, Flux.identity),  # μ
        Dense(256, latent_dim, Flux.identity)   # log(σ)
    )

    # Define decoder: latent → dense layers → reconstruction
    decoder = AutoEncoderToolkit.BernoulliDecoder(Flux.Chain(
        Dense(latent_dim, 256, identity),
        Dense(256, 400, relu),
        Dense(400, input_dim, sigmoid)
    ))

    # === 3. Set up VAE and optimizer ===
    vae = encoder * decoder

    # Move to GPU if available
    if CUDA.has_cuda()
        println("Using GPU for training.")
        vae = vae |> gpu
        train_data = train_data |> gpu
        val_data = val_data |> gpu
    else
        println("CUDA not available. Running on CPU.")
    end

    # Adam optimizer
    opt_vae = Flux.Train.setup(Flux.Optimisers.Adam(1e-3), vae)

    # === 4. Create DataLoader ===
    train_loader = Flux.DataLoader(train_data, batchsize=10, shuffle=true)

    # === 5. Initialize metrics ===
    train_loss = zeros(Float32, num_epochs)
    val_loss = zeros(Float32, num_epochs)
    train_entropy = zeros(Float32, num_epochs)
    val_entropy = zeros(Float32, num_epochs)
    train_mse = zeros(Float32, num_epochs)
    val_mse = zeros(Float32, num_epochs)

    # === 6. Training loop ===
    for epoch in 1:num_epochs
        println("Epoch: $epoch")

        for (i, x) in enumerate(train_loader)
            x = CUDA.has_cuda() ? x |> gpu : x  # Move batch to GPU if available
            println("Epoch $epoch | Batch $i / $(length(train_loader))")
            AutoEncoderToolkit.VAEs.train!(vae, x, opt_vae)
        end

        # === 7. Validation and Metrics ===
        train_loss[epoch] = AutoEncoderToolkit.VAEs.loss(vae, train_data)
        val_loss[epoch] = AutoEncoderToolkit.VAEs.loss(vae, val_data)

        train_outputs = vae(train_data)
        val_outputs = vae(val_data)

        train_entropy[epoch] = Flux.Losses.logitbinarycrossentropy(train_outputs.p, train_data)
        val_entropy[epoch] = Flux.Losses.logitbinarycrossentropy(val_outputs.p, val_data)

        train_mse[epoch] = Flux.mse(train_outputs.p, train_data)
        val_mse[epoch] = Flux.mse(val_outputs.p, val_data)

        # === 8. Report progress ===
        println("Epoch $epoch / $num_epochs:")
        println("- Train MSE: $(train_mse[epoch])")
        println("- Val MSE: $(val_mse[epoch])")
        println("- Train Loss: $(train_loss[epoch])")
        println("- Val Loss: $(val_loss[epoch])")
        println("- Train Entropy: $(train_entropy[epoch])")
        println("- Val Entropy: $(val_entropy[epoch])")
    end

    # === 9. Return model and metrics ===
    return vae, (
        train_loss=train_loss, val_loss=val_loss,
        train_mse=train_mse, val_mse=val_mse,
        train_entropy=train_entropy, val_entropy=val_entropy
    )
end
rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())


function run_pca(connectivity_matrix::Matrix{Float64}; dims::Int=2)
    # Normalize each feature to [0, 1]
    connectivity_matrix = rescale(connectivity_matrix, dims=1)
    
    # Run PCA
    pca_model = fit(PCA, connectivity_matrix; maxoutdim=dims)
    pca_result = transform(pca_model, connectivity_matrix)
    
    return pca_result
end
function run_tsne(connectivity_matrix::Matrix{Float64}; dims::Int=2, perplexity::Int=30)
    connectivity_matrix = rescale(connectivity_matrix, dims=1)  # Normalize each feature to [0, 1]
    
    tsne_result = tsne(connectivity_matrix, dims,  0,   1000 , perplexity)
    return tsne_result
end


function cluster(matrix_to_cluster::Matrix{Float64}, 
                      group_array::Vector{Int}, 
                      method::String, 
                      cluster_map::Dict{Int, Int})

"""

# Arguments
- matrix_to_cluster: Matrix (subjects x features)
- group_array: Vector of true group numbers (e.g., [1, 2, 3, 4, ...])
- method: "kmeans" or "gmm"
- cluster_map: Dict mapping group numbers to desired cluster indices (e.g., Dict(4=>2, 1=>1, 2=>1, 3=>1))

# Returns
- pred_labels: predicted cluster assignments
- labels: ground-truth cluster assignments as defined in `cluster_map`
"""
    # Map ground truth labels into desired cluster indices
    labels = map(x -> get(cluster_map, x, missing), group_array)

    if any(ismissing, labels)
        error("Some group values in group_array are not defined in cluster_map.")
    end

    labels = collect(labels)
    num_clusters = length(unique(values(cluster_map)))

    # Run clustering
    pred_labels = nothing
    if method == "kmeans"
        kmeans_result = kmeans(matrix_to_cluster', num_clusters; maxiter=300, display=:none)
        pred_labels = kmeans_result.assignments
    elseif method == "fuzzy"
        # Apply fuzzy C-means on matrix_to_cluster (shape: n_points × dim)
        fuzzy_result = fuzzy_cmeans(matrix_to_cluster', num_clusters, 2.0; maxiter=300, display=:none)
        
        # Extract soft membership weights (n_points × num_clusters)
        memberships = fuzzy_result.weights

        # Convert soft assignment to hard labels by choosing the max-weighted cluster
        pred_labels = map(i -> argmax(memberships[i, :]), 1:size(memberships, 1))
    elseif method == "kmedoids"
        # Step 1: Compute pairwise distance matrix
        dist_matrix = pairwise(Euclidean(), matrix_to_cluster, dims=1)

        # Step 2: Run K-medoids on the distance matrix
        kmedoids_result = kmedoids(dist_matrix, num_clusters; init=:kmpp, maxiter=300, display=:none)

        # Step 3: Extract cluster assignments
        pred_labels = kmedoids_result.assignments
    
    elseif method == "hierarchical"
        # Step 1: Compute pairwise distance matrix
        dist_matrix = pairwise(Euclidean(), matrix_to_cluster, dims=1)

        # Step 2: Perform hierarchical clustering
        hclust_result = hclust(dist_matrix, linkage=:ward)

        # Step 3: Cut dendrogram into desired number of clusters
        pred_labels = cutree(hclust_result, k=num_clusters)
    
    elseif method == "mcl"

        # Step 1: Compute distance matrix
        dist_matrix = pairwise(Euclidean(), matrix_to_cluster, dims=1)

        # Step 2: Convert distances to similarities via Gaussian kernel
        σ = quantile(dist_matrix[dist_matrix .> 0], 0.1)  # More sensitive σ
        sim_matrix = exp.(-dist_matrix.^2 / (2 * σ^2))

        # Step 3: Add self-loops manually if desired (though `add_loops=true` should handle it)
        for i in 1:size(sim_matrix, 1)
            sim_matrix[i, i] = 1.0
        end
        sim_matrix = sim_matrix .+ 1e-6*rand(size(sim_matrix)...)

        # Step 4: Run MCL algorithm
        mcl_result = mcl(sim_matrix;
            add_loops=false,  # already manually added
            expansion=2,
            inflation=2.5,     # Try values from 1.2 to 2.5 for more/less granularity
            maxiter=100,
            tol=1e-6,
            save_final_matrix=true
        )

        # Step 5: Get cluster labels
        pred_labels = mcl_result.assignments



    elseif method == "dbscan"
        # DBSCAN parameters
        eps = 0.5              # radius for neighborhood
        min_pts = 5            # minimum neighbors to form a core point
        min_cluster_size = 5   # minimum number of points in a cluster

        # matrix_to_cluster must be d × n for DBSCAN (each column is a point)
        dbscan_result = dbscan(matrix_to_cluster', eps;
                            metric=Euclidean(),
                            min_neighbors=min_pts,
                            min_cluster_size=min_cluster_size)

        # Cluster assignments: 0 means noise/unassigned
        pred_labels = dbscan_result.assignments
    
    elseif method == "affinityprop"
        # Compute similarity matrix from matrix_to_cluster
        # Convert distances to similarities (negative Euclidean distances)
        D = pairwise(Euclidean(), matrix_to_cluster', matrix_to_cluster')
        S = -D

        # Optional: set diagonal (preferences) to median similarity
        median_sim = median(S[.~I])  # I is the identity mask to exclude diagonal
        for i in 1:size(S, 1)
            S[i, i] = median_sim
        end

        # Run Affinity Propagation
        affprop_result = affinityprop(S; maxiter=200, tol=1e-6, damp=0.5)

        # Cluster assignments
        pred_labels = affprop_result.assignments
     
    elseif method == "gmm"

        gmm = GMM(num_clusters, matrix_to_cluster; method=:kmeans, kind=:full, nIter=50)
        #em!(gmm, matrix_to_cluster) 

        # Predict cluster assignments
        loglik = avll(gmm, matrix_to_cluster)
        #println("Average Log Likelihood = ", loglik)
        posterior_probs, _ = gmmposterior(gmm, matrix_to_cluster)
        pred_labels = map(i -> argmax(posterior_probs[i, :]), 1:size(matrix_to_cluster, 1))

    else
        error("Invalid clustering method: choose 'kmeans' or 'gmm'")
    end

    return pred_labels, labels
end



# Define the result structure outside the function so it's available to callers
struct RunResult
    data_name::String
    method::String
    dims::Int
    clustering::String
    mean_acc::Float64
    std_acc::Float64
    predictions::Vector{Vector{Int}}  # Stores predictions for each repeat
    true_labels::Vector{Vector{Int}}  # Stores true labels for each repeat
    exp_data::Vector{Matrix{Float64}} # Stores the embedded data for each repeat
end

function evaluate_dimensionality_reduction(
    dims_list::Vector{Int},
    data_sources::Dict{String, Matrix{Float64}},
    methods::Vector{String},
    repeats::Int,
    clustering_methods::Vector{String},
    vae_results::Dict,
    group_array::Vector{Int},
    select_counts::Vector{Int}
)::Tuple{DataFrame, GroupedDataFrame, Vector{RunResult}}
    # === evaluate_dimensionality_reduction ===
    # Evaluates multiple dimensionality reduction and clustering methods on connectivity data.
    # 
    # Inputs:
    # - dims_list: List of latent dimensions to test (e.g., [2, 10, 50])
    # - data_sources: Dict of datasets (e.g., "RDCM", "FC") → subject × feature matrices
    # - methods: Dimensionality reduction methods (e.g., ["PCA", "tSNE", "VAE"])
    # - repeats: Number of repetitions for each configuration
    # - clustering_methods: Clustering algorithms (e.g., ["kmeans", "fuzzy", "hierarchical"])
    # - vae_results: Pretrained VAE models for each dataset/dimension
    # - group_array: Diagnostic labels for all subjects
    # - select_counts: How many subjects to sample per group
    #
    # Outputs:
    # - results: Summary DataFrame with mean/std accuracy for each config
    # - grouped: Grouped DataFrame by dataset, method, clustering
    # - all_results: List of RunResult structs with detailed metrics and embeddings
    #
    # RunResult struct holds:
    # - data_name, method, dims, clustering
    # - mean_acc, std_acc
    # - predictions, true_labels, exp_data (for each repeat)

    # Initialize a list to store all results
    all_results = RunResult[]
    
    # Initialize DataFrame for summary results
    results = DataFrame(
        Data=String[], Method=String[], Dims=Int[], 
        Clustering=String[], MeanAcc=Float64[], StdAcc=Float64[]
    )

    for (data_name, matrix) in data_sources
        for dims in dims_list
            for method in methods
                for clustering in clustering_methods
                    acc_list = Float64[]
                    pred_list = Vector{Int}[]
                    true_list = Vector{Int}[]
                    data_list = Matrix{Float64}[]
                    
                    for i in 1:repeats
                        # Dimensionality reduction
                        if method == "PCA"
                            pca_result = run_pca(Float64.(transpose(matrix)); dims=dims)
                            μ_result = Float64.(transpose(pca_result))
                        elseif method == "tSNE"
                            μ_result = run_tsne(matrix; dims=dims, perplexity=40)
                        elseif method == "VAE"
                            vae_model = vae_results[data_name][string(dims)][:model]
                            vae_latent = vae_model.encoder(Float64.(transpose(matrix|> gpu)))
                            μ_result = Float64.(transpose(vae_latent.μ |> cpu))
                        elseif method == "NoLatent"
                            μ_result = Float64.(matrix)
                        else
                            error("Unknown method: $method")
                        end

                        # Subject selection
                        selected_data, selected_labels, _ = select_subjects_by_group(
                            μ_result, group_array, select_counts
                        )
                        exp_data = selected_data
                        labels = selected_labels

                        # Clustering and evaluation
                        pred, true_labels = cluster(
                            exp_data, 
                            labels, 
                            clustering, 
                            Dict(1=>1, 2=>2, 4=>3)
                        )
                        
                        acc = accuracy_score(true_labels, pred)
                        
                        push!(acc_list, acc)
                        push!(pred_list, pred)
                        push!(true_list, true_labels)
                        push!(data_list, exp_data)
                    end

                    # Calculate summary statistics
                    mean_acc = mean(acc_list)
                    std_acc = std(acc_list)
                    
                    # Store complete results
                    push!(all_results, RunResult(
                        data_name, method, dims, clustering,
                        mean_acc, std_acc,
                        pred_list, true_list, data_list
                    ))
                    
                    # Store summary in DataFrame
                    push!(results, (
                        data_name, method, dims, clustering, 
                        mean_acc, std_acc
                    ))
                end
            end
        end
    end

    # Group the DataFrame results
    grouped = groupby(results, [:Data, :Method, :Clustering])
    
    return results, grouped, all_results
end

# Accuracy using label matching
function accuracy_score(true_labels::Vector{Int}, pred_labels::Vector{Int})
    u_true = unique(true_labels)
    u_pred = unique(pred_labels)
    length(u_true) != length(u_pred) && return 0.0
    

    best = 0.0
    for p in permutations(u_pred)
        # Create a mapping dictionary
        mapping = Dict(zip(u_pred, p))
        # Remap predictions
        remapped = [mapping[l] for l in pred_labels]
        # Compute accuracy
        acc = mean(remapped .== true_labels)
        best = max(best, acc)
    end
    return best
end