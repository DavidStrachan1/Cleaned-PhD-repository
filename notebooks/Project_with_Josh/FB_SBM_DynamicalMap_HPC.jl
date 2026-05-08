using ITensors, ITensorMPS, LinearAlgebra, Plots, JLD2, CSV, DataFrames, PolyChaos
BLAS.set_num_threads(1) # avoid multi-threading issues with BLAS in correlation matrix calculation

# Define a trivial qubit space for QN conservation
function ITensors.space(::SiteType"QubitWithQN";
    conserve_qns::Bool=false, 
    qnname::String="Q")
    conserve_qns ? [QN(qnname, 0) => 2] : 2    
end

# Define states for qubit with QN
function ITensors.state(::StateName{name}, ::SiteType"QubitWithQN", s::Index) where {name}
    nm = string(name)
    if nm == "0"
        return onehot(s => 1)
    elseif nm == "1"
        return onehot(s => 2)
    else
        error("Unknown state name: $name, use 0 or 1.")
    end
end

# Define operators for qubit with QN
ITensors.op(::OpName"Id", ::SiteType"QubitWithQN") = [1 0; 0 1]
ITensors.op(::OpName"Z",  ::SiteType"QubitWithQN") = [1 0; 0 -1]
ITensors.op(::OpName"X",  ::SiteType"QubitWithQN") = [0 1; 1 0]
ITensors.op(::OpName"Y",  ::SiteType"QubitWithQN") = [0 -im; im 0]

# Define "balancing" (shift) operators for conserving QN via ancilla
function ITensors.op(::OpName"Beta", ::SiteType"Boson", d::Int)
    M = zeros(d, d)
    for n in 1:d-1
        M[n+1, n] = 1.0
    end
    return M
end

function ITensors.op(::OpName"BetaDag", ::SiteType"Boson", d::Int)
    M = zeros(d, d)
    for n in 2:d
        M[n-1, n] = 1.0
    end
    return M
end

struct SimParams
    L::Int # Total length of bath chain, not including impurity.
    dt::Float64 # Time step
    nsteps::Int # Number of time steps to evolve
    wc::Float64 # exponential cutoff frequency, for using with exp-ohmic
    alpha::Float64 # coupling strength
    epsilon_sigmaz::Float64 # impurity energy splitting
    delta::Float64 # sigma_x term strength
    beta::Float64 # inverse temperature 
    s::Float64 # ohmic bath exponent

    probabilityBosonDim::Float64 # threshold for increasing local dim
    epsilon::Float64 # tolerance for empty/full
    apply_cutoff::Float64 # cutoff for applying truncation
    time_evolution_cutoff::Float64 # cutoff for time evolution
    initDim::Int # initial local dimension for bath sites, can be increased during simulation if needed
    save_results::Bool # whether to save results to file
end

mutable struct SimState
    psi::MPS
    F::Matrix{ComplexF64} # rotating frame
    active::Vector{Int} # active bath sites in rotating frame
    step::Int
    t::Float64
    ni::Vector{Float64} # occupation of each mode, cheaper than handling full cc
    cc_active::Matrix{ComplexF64} # correlation matrix of active subspace
    sites::Vector{Index} # site indices of the MPS, needed for local dimension update

    # Measurements:
    times::Vector{Float64} # time points for measurements
    dynamicalmaps::Array{ComplexF64,3} # store dynamical maps at each time step
    propogators::Array{ComplexF64,3} # store propogators at each time step
end

mutable struct Gate
    modes::Vector{Int}
    c::ComplexF64
    s::ComplexF64
end

function boseEinsteinFunction(beta, omega)
    return 1.0/expm1(beta * omega)
end

function heavisideFunction(x)
    return x > 0 ? 1.0 : 0.0
end

function bathSpectralFunction(p::SimParams)
    J = (ω::Float64) -> 2.0 * p.alpha * p.wc^(1 - p.s) * ω^p.s * exp(-ω / p.wc)
    supp = (-10.0 * p.wc, 10.0 * p.wc)
    return J, supp
end

function generateTTEDOPACoefficients(p::SimParams)
    J, supp = bathSpectralFunction(p)
    Jthermal(x) = begin 
        wabs = abs(x)
        Jv = J(wabs)
        nv = boseEinsteinFunction(p.beta, wabs)
        Jv * (nv + heavisideFunction(x))
    end

    if p.L > 100
        @warn "L is too large to calculate all T-TEDOPA coeffs - assumed to be constant after 100 coeffs."
        coeffLength = 100
    else
        coeffLength = p.L
    end


    wmin, wmax = supp[1], supp[2]
    function makequad(a,b,wfun)
        return N -> begin
            xbar, wbar = fejer(N)
            x = ((b-a)/2) .* xbar .+ ((b+a)/2)
            w = (b-a)/2 .* wbar .* wfun.(x)
            return x, w
        end
    end

    w_left(x) = Jthermal(x)
    w_right(x) = Jthermal(x)

    tol=1E-2
    delta=1E-6
    delta_min = 1E-12
    alpha, beta, alpha_old, beta_old = nothing, nothing, nothing, nothing
    


    quad_left = makequad(wmin, -delta, w_left)
    quad_right = makequad(delta, wmax, w_right)
    while true
        quad_left = makequad(wmin, -delta, w_left)
        quad_right = makequad(delta, wmax, w_right)
        quads = [quad_left, quad_right]
        alpha_new, beta_new = mcdiscretization(coeffLength, quads; discretization=lanczos, Nmax=10000000, ε=1E-8)
        if alpha_old !== nothing 
            maxdiff_alpha, maxdiff_beta = maximum(abs.(alpha_new .- alpha_old)), maximum(abs.(sqrt.(beta_new) .- sqrt.(beta_old)))
            println("Max difference in alpha: ", maxdiff_alpha, ", Max difference in beta: ", maxdiff_beta, " with delta: ", delta)
            if maxdiff_alpha <= tol && maxdiff_beta <= tol
                alpha, beta = alpha_new, beta_new
                break
            end
        end
        alpha_old_true, beta_old_true = alpha_old, beta_old
        alpha_old, beta_old = alpha_new, beta_new
        delta = delta/2
        if delta < delta_min
            @warn "Delta reached minimum value without convergence, results may not be fully converged!"
            alpha, beta = alpha_new, beta_new
            break
        end
    end


    if p.L > 100
        alpha_actual = zeros(p.L)
        alpha_actual[1:length(alpha)] .= alpha
        alpha_actual[length(alpha)+1:end] .= alpha[end]

        beta_actual = zeros(p.L)
        beta_actual[1:length(beta)] .= beta
        beta_actual[length(beta)+1:end] .= beta[end]
    else
        alpha_actual = alpha
        beta_actual = beta
    end



    return alpha_actual, sqrt.(beta_actual)
end

function generateDiscretisedBathCoefficients(p::SimParams)
    wmax = 10 * p.wc
    Omega = collect(range(0.0, wmax; length=p.L+1))
    w = zeros(Float64, p.L)
    g = zeros(Float64, p.L)  
    for k in 1:p.L
        g[k] = 2 * p.alpha * p.wc * (-exp(-Omega[k+1]/p.wc)*(p.wc+Omega[k+1]) + exp(-Omega[k]/p.wc)*(p.wc+Omega[k]))
        w[k] = -2 * p.alpha * p.wc * (exp(-Omega[k+1]/p.wc) * (2*p.wc^2 + 2*p.wc*Omega[k+1] + Omega[k+1]^2) - exp(-Omega[k]/p.wc) * (2*p.wc^2 + 2*p.wc*Omega[k] + Omega[k]^2))/g[k]
    end
    g = sqrt.(g)
    return w, g
end

function generateHamiltonianCoefficients(p::SimParams)
    E, J = generateTTEDOPACoefficients(p)
    Hb = zeros(ComplexF64, p.L, p.L)
    for i in 1:p.L-1
        Hb[i,i+1] = J[i+1]
        Hb[i+1,i] = J[i+1]
    end
    for i in 1:p.L
        Hb[i,i] = E[i]
    end
    eta = zeros(ComplexF64, p.L)
    eta[1] = J[1]

    return Hb, eta
end

function increaseLocalDimensionQN!(st::SimState, p::SimParams, sites, ni_active)
    active_modes = st.active
    sites = siteinds(st.psi)

    bath0 = 4 # first bath site
    anc = 3 # ancilla for first bath site 

    # temporarily add a site to active mode
    if isempty(active_modes)
        push!(active_modes, 1)
    else
        next = active_modes[end] + 1
        if next <= p.L
            push!(active_modes, next)
        end
    end

    prob_n(rho::ITensor, s::Index, n::Int) = real((dag(onehot(prime(s) => n)) * rho * onehot(s => n))[])

    # compute the probability of maximum occupation for each active mode
    boson_dimensions = fill(1, p.L)

    
    for i in active_modes
        j = 3 + i   # true index of active modes in the MPS (2 system, 1 bath ancilla)
        boson_dimensions[i] = ITensors.dim(sites[j])
        s = sites[j]
        orthogonalize!(st.psi, j)
        nmax = ITensors.dim(s)

        rhoj = prime(st.psi[j], s) * dag(st.psi[j])
        rhojM = Array(rhoj, prime(s), dag(s))
        probs = real.(diag(rhojM))

        prob = probs[end]
        for x in 1:3
            if length(probs) > x
                prob += probs[end-x] # also consider probability of second highest occupation, to avoid getting stuck at dim=2 if dim=3 is actually needed.
            end
        end


        if prob > p.probabilityBosonDim
            boson_dimensions[i] += 2
        end
    end

    psi = copy(st.psi)
    orthogonalize!(psi, min(2, length(psi)))
    for i in 1:p.L
        oldsite = sites[3 + i]
        boson_dimensions[i] = max(boson_dimensions[i], ITensors.dim(oldsite)) # ensure increases only
    end
    
    # Ancilla dim must match bath mode 1
    anc_dim = boson_dimensions[1]

    # construct the site inds: [impurities..., ancilla, bath...]
    newsiteinds_bath = [siteind("Boson"; conserve_qns=true, dim=d) for d in boson_dimensions]
    newsiteinds = vcat(
        [siteind("QubitWithQN"; conserve_qns=true) for _ in 1:2]...,
        siteind("Boson"; conserve_qns=true, dim=anc_dim),
        newsiteinds_bath...
    )

    # embed old MPS into new sites
    N = 3 + p.L
    psiExpanded = Vector{ITensor}(undef, N)

    for j in 1:2
        psiExpanded[j] = psi[j]
    end

    begin
        A = psi[anc]
        oldsite = sites[anc]
        newsite = newsiteinds[anc]

        if oldsite == newsite
            psiExpanded[anc] = A
        else
            inds_A = collect(inds(A))
            k = findfirst(==(oldsite), inds_A)
            inds_A[k] = newsite

            A_pad = ITensor(inds_A...)
            for n in 1:ITensors.dim(oldsite)
                A_pad += (A * dag(onehot(oldsite => n))) * onehot(newsite => n)
            end
            psiExpanded[anc] = A_pad
        end
    end

    for i in 1:p.L
        j = 3 + i
        A = psi[j]

        oldsite = sites[j]
        newsite = newsiteinds[j]

        if oldsite == newsite
            psiExpanded[j] = A
            continue
        end

        inds_A = collect(inds(A))
        k = findfirst(==(oldsite), inds_A)
        inds_A[k] = newsite

        A_pad = ITensor(inds_A...)
        for n in 1:ITensors.dim(oldsite)
            A_pad += (A * dag(onehot(oldsite => n))) * onehot(newsite => n)
        end

        psiExpanded[j] = A_pad
    end

    st.psi = MPS(psiExpanded)
    orthogonalize!(st.psi, min(2, length(st.psi)))
    return siteinds(st.psi)
end

function initializeSimState(p::SimParams)
    chainLength = p.L + 3
    sites = siteinds("Boson", chainLength; conserve_qns=true, dim=1)
    sites[1] = siteind("QubitWithQN"; conserve_qns=true)
    sites[2] = siteind("QubitWithQN"; conserve_qns=true)
    active_dims = 11 # start with dim=11 for first few modes to allow for active growth
    for j in 3:8
        sites[j] = siteind("Boson"; conserve_qns=true, dim=active_dims)
    end
    # and make sure site 3,4 have a large initial dim, for num conservation
    initdim = p.initDim
    sites[3] = siteind("Boson"; conserve_qns=true, dim=initdim)
    sites[4] = siteind("Boson"; conserve_qns=true, dim=initdim)
    states = ["0" for i in 1:chainLength] # initial vac state
    states[3] = string(initdim - 1) # ancilla maximally occupied.
   

    psi = MPS(sites, states)

    # Form the initial Choi state:
    s1 = sites[1] # Choi site
    s2 = sites[2] # system site
    link = linkind(psi,2) # link between system-bath

    phi12 = ITensor(s1, s2, link)
    phi12[s1 => 1, s2 => 1, link => 1] = 1 / sqrt(2)
    phi12[s1 => 2, s2 => 2, link => 1] = 1 / sqrt(2)
    U, S, V = svd(phi12, s1)
    psi[1] = U
    psi[2] = S * V

    F = Matrix{ComplexF64}(I, p.L, p.L) # start in non-rotating frame


    cc = zeros(ComplexF64, p.L, p.L)
    ni = real.(diag(cc))
    active = findall((ni .> p.epsilon))

    sites = siteinds(psi)

    st = SimState(psi, F, active, 0, 0.0, zeros(Float64, p.L), zeros(Float64, p.L, p.L), sites, Float64[], zeros(ComplexF64, 4, 4, p.nsteps), zeros(ComplexF64, 4, 4, p.nsteps))
    return st
end

function givens_rotation(v_in::AbstractVector{ComplexF64}, modes::AbstractVector{<:Integer})
    m = length(modes)
    v = copy(v_in)
    G_subsys = Matrix{ComplexF64}(I, m, m)
    gates = Gate[]
    sizehint!(gates, max(m - 1, 0))

    if m <= 1
        return G_subsys, gates
    end

    for j in (m - 1):-1:1
        a = v[j]
        b = v[j + 1]
        r2 = abs2(a) + abs2(b)

        if r2 > 1e-30
            r  = sqrt(r2)
            aN = a / r
            bN = b / r
            c  = abs(aN)
            s  = (c > 1e-15) ? (aN * conj(bN) / c) : (-conj(bN))
        else
            c = 1.0
            s = 0.0 + 0.0im
        end

        push!(gates, Gate([Int(modes[j]), Int(modes[j + 1])], ComplexF64(c), s))

        vj  = v[j]
        vjp = v[j + 1]
        v[j]     = c * vj + s * vjp
        v[j + 1] = -conj(s) * vj + c * vjp

        for col in 1:m
            x = G_subsys[j, col]
            y = G_subsys[j + 1, col]
            G_subsys[j, col]     = c * x + s * y
            G_subsys[j + 1, col] = -conj(s) * x + c * y
        end
    end

    return G_subsys, gates
end

function apply_givens_left!(x::AbstractVector{ComplexF64}, g::Gate)
    i, j = g.modes
    xi, xj = x[i], x[j]
    x[i] = g.c * xi + g.s * xj
    x[j] = -conj(g.s) * xi + g.c * xj
    return x
end

function apply_givens_left!(x::AbstractVector{ComplexF64}, gates::AbstractVector{Gate})
    for g in gates
        apply_givens_left!(x, g)
    end
    return x
end

function apply_givens_left!(M::AbstractMatrix{ComplexF64}, g::Gate)
    i, j = g.modes
    @views begin
        ri = copy(M[i, :])
        rj = copy(M[j, :])
        M[i, :] .= g.c .* ri .+ g.s .* rj
        M[j, :] .= -conj(g.s) .* ri .+ g.c .* rj
    end
    return M
end

function apply_givens_left!(M::AbstractMatrix{ComplexF64}, gates::AbstractVector{Gate})
    for g in gates
        apply_givens_left!(M, g)
    end
    return M
end

function apply_givens_right_dag!(M::AbstractMatrix{ComplexF64}, g::Gate)
    i, j = g.modes
    @views begin
        ci = copy(M[:, i])
        cj = copy(M[:, j])
        M[:, i] .= g.c .* ci .+ conj(g.s) .* cj
        M[:, j] .= -g.s .* ci .+ g.c .* cj
    end
    return M
end

function apply_givens_right_dag!(M::AbstractMatrix{ComplexF64}, gates::AbstractVector{Gate})
    for g in gates
        apply_givens_right_dag!(M, g)
    end
    return M
end

function apply_givens_similarity!(C::AbstractMatrix{ComplexF64}, g::Gate)
    apply_givens_left!(C, g)
    apply_givens_right_dag!(C, g)
    return C
end

function apply_givens_similarity!(C::AbstractMatrix{ComplexF64}, gates::AbstractVector{Gate})
    for g in gates
        apply_givens_similarity!(C, g)
    end
    return C
end

function remap_gates(gates_local::AbstractVector{Gate}, labels::AbstractVector{<:Integer})
    gates = Gate[]
    sizehint!(gates, length(gates_local))
    for g in gates_local
        push!(gates, Gate([Int(labels[g.modes[1]]), Int(labels[g.modes[2]])], g.c, g.s))
    end
    return gates
end

function localiseHc2Rot!(st::SimState, p::SimParams, eta_tc_rot::Vector{ComplexF64}, ni_bath::Vector{Float64})

    gates_emp = Gate[]
    nActive = length(st.active)

    if nActive >= p.L
        println("All modes active, skipping localisation.")
        return gates_emp, eta_tc_rot
    end

    tail = (nActive + 1):p.L
    empty = findall(x -> x < 0.5, view(ni_bath, tail)) .+ nActive
    couplings = eta_tc_rot[empty]

    if !isempty(empty) && norm(couplings) > p.epsilon
        _, gates_emp = givens_rotation(couplings, empty)

        # exact replacement for eta_tc_rot = G_emp * eta_tc_rot
        apply_givens_left!(eta_tc_rot, gates_emp)

        push!(st.active, empty[1])
        sort!(st.active)

    end

    return gates_emp, eta_tc_rot
end

function applyGivens!(st::SimState, p::SimParams, givens_gates, sites, L)
    for g in givens_gates

        s1, s2 = g.modes[1], g.modes[2]

        s1 += 3
        s2 += 3


        if s1 + 1 != s2
            println("s1: ", s1, " s2: ", s2)
            println("MPS Active:", st.active)
            error("Givens rotation can only be applied to adjacent modes.")
        end
            
            s1s, s2s = siteind(st.psi, s1), siteind(st.psi, s2)

            c = clamp(real(g.c), 0, 1.0)
            theta = acos(c)
            phi = (abs(g.s) > 0) ? angle(g.s) : 0.0
            alpha = theta * cis(phi)
            K = alpha * op("Adag", s1s) * op("A", s2s) - conj(alpha) * op("Adag", s2s) * op("A", s1s)
            U = exp(K)
     
            st.psi = apply(U, st.psi; cutoff=p.apply_cutoff, maxdim=1E10) 

            st.sites = siteinds(st.psi) # update sites after potential change in optimal basis

    end
    normalize!(st.psi)
    return nothing
end

function calculateCorrelationMatrix(st::SimState, p::SimParams, sites; init_corr_calc=false)
        cc = Matrix{ComplexF64}(correlation_matrix(st.psi, "Adag", "A"; sites=sites))
    return cc
end

function calculateDensity(st::SimState, p::SimParams)
        d = expect(st.psi, "N"; sites=4 : 3 + p.L)
    return d
end

function timeEvolve!(st::SimState, p::SimParams, eta_tc_rot, sites)
   
    ni_bath = st.ni
    increaseLocalDimensionQN!(st, p, siteinds(st.psi), ni_bath)

    
    dims = [Int(ITensors.dim(siteind(st.psi, i))) for i in 1:length(siteinds(st.psi))]
    
    op = OpSum()
    op += 0.5*p.epsilon_sigmaz, "Z", 2
    op += 0.5*p.delta, "X", 2
    op += eta_tc_rot[1], "Z", 2, "BetaDag", 3, "Adag", 4
    op += conj(eta_tc_rot[1]), "Z", 2, "Beta", 3, "A", 4

    H = MPO(op, siteinds(st.psi))

    st.psi = expand(st.psi, H; alg="global_krylov", krylovdim=2, cutoff=1E-6, apply_kwargs=(; maxdim=3000, cutoff=1E-10))
    st.psi = tdvp(H, -1im * p.dt, st.psi; cutoff=p.time_evolution_cutoff, nsteps=1, outputlevel=1, nsite=1, normalize=true)
   
 
    st.sites = siteinds(st.psi) # update sites after potential change in optimal basis
    @show maxlinkdim(st.psi)

    return sites
end

function active_diagonalization(st::SimState, p::SimParams)
    nactive = length(st.active)

    if nactive == 0
        return Matrix{ComplexF64}(I, p.L, p.L), Gate[], ComplexF64[]
    end

    gates_nat = Gate[]
    sizehint!(gates_nat, nactive * (nactive - 1) ÷ 2)

    V = Matrix{ComplexF64}(I, p.L, p.L)

    first_active = minimum(st.active) + 3
    last_active = maximum(st.active) + 3
    sites=first_active:last_active
    cc_active = calculateCorrelationMatrix(st, p, sites)
    rho_sub = Matrix(transpose(cc_active)) # need to transpose to match the convention of rho_sub = V_col * rho_sub * V_col' where V_col acts on columns of rho_sub. This is because correlation_matrix returns <A_i A_j^dag> = (V_col * rho_sub * V_col')[i,j] = sum_k V_col[i,k] * rho_sub[k,l] * conj(V_col[j,l]) which has the same structure as a similarity transform if we take V_col to be the matrix that acts on the columns of rho_sub, i.e. V_col[k,l] instead of V_col[l,k].

    for j in 1:nactive
        block = Matrix(@view rho_sub[j:nactive, j:nactive])
        eig = eigen(Hermitian(block))
        vals, vecs = eig.values, eig.vectors

        activity(x) = min(abs(x), abs(1 - x))
        #most_active_indx = argmax(activity.(vals))
        most_active_indx = argmax(vals) # just diagonalize in order of occupation, to avoid issues with near-degeneracies and mode swapping
        vec = vecs[:, most_active_indx]

        # local gates act on rho_sub indices j:nactive
        _, gates_local = givens_rotation(vec, j:nactive)

        # remap those same gates to bath-mode labels for V / later use
        gates_bath = remap_gates(gates_local, st.active)

        append!(gates_nat, gates_bath)

        # exact replacement for V = V_col * V
        apply_givens_left!(V, gates_bath)

        # exact replacement for rho_sub = V_col * rho_sub * V_col'
        apply_givens_similarity!(rho_sub, gates_local)
    end

    # Mathematically identical to diag((V * st.cc * V')[st.active, st.active])
    population = diag(rho_sub)

    return V, gates_nat, population
end

function reduceActive!(st::SimState, p::SimParams, pop)
    old_active = copy(st.active)
    inds = findall(real.(pop) .> p.epsilon) 
    st.active=st.active[inds]
    sort!(st.active)
   
    return nothing
end

function measure!(st::SimState, p::SimParams)
    psi = copy(st.psi)
    push!(st.times, st.t)

    st.sites = siteinds(st.psi) # update sites after potential change in optimal basis

    dynamical_map = calculateDynamicalMap(st)
    st.dynamicalmaps[:, :, st.step] = dynamical_map

    if st.step != 1
        propogator = (st.dynamicalmaps[:, :, st.step] - st.dynamicalmaps[:, :, st.step - 1]) / p.dt
        propogator = propogator * pinv(st.dynamicalmaps[:,:, st.step])
        st.propogators[:, :, st.step] = propogator
    end
    return nothing
end

function evolve_F_left!(F::Matrix{ComplexF64}, ub_phase::AbstractVector{ComplexF64})
    @inbounds for i in axes(F, 1)
        @views F[i, :] .*= ub_phase[i]
    end
    return F
end

function evolve_F_left!(F::Matrix{ComplexF64}, Ub::Matrix{ComplexF64})
    Fold = copy(F)
    mul!(F, Ub, Fold)
    return F
end

function calculateDynamicalMap(st::SimState)
    orthogonalize!(st.psi, 2)

    s1 = siteind(st.psi, 1)   # reference qubit
    s2 = siteind(st.psi, 2)   # physical system qubit

    θ = st.psi[1] * st.psi[2]
    ρAS = prime(θ, s1, s2) * dag(θ) # reduced density matrix of system and reference, with primed indices for output legs

    T = Array(dense(ρAS), prime(s1), prime(s2), dag(s1), dag(s2)) # reshape into 4-leg tensor with legs in order (s1', s2', s1, s2)

    Λ = zeros(ComplexF64, 4, 4)

    for i in 1:2, j in 1:2
        col = lin(i, j)

        Eij = zeros(ComplexF64, 2, 2)
        for m in 1:2, n in 1:2
            Eij[m, n] = 2 * T[i, m, j, n]
        end

        Λ[:, col] = vec(Eij)
    end
end
function step!(st::SimState, p::SimParams, Ub, eta_tc)
    eta_tc_rot = st.F' * eta_tc

    ni_bath = st.ni

    gates_emp, eta_tc_rot = localiseHc2Rot!(st, p, eta_tc_rot, ni_bath)

    coupling = eta_tc_rot[st.active]
    _, gates_active = givens_rotation(coupling, st.active)

    # exact replacement for eta_tc_rot = G_active * eta_tc_rot
    apply_givens_left!(eta_tc_rot, gates_active)
    
    applyGivens!(st, p, gates_active, st.sites, p.L)

    st.sites = timeEvolve!(st, p, eta_tc_rot, st.sites)

    # calculate on-site densities
    st.ni = calculateDensity(st, p)
   
    V_nat, gates_nat, pop = active_diagonalization(st, p)

    applyGivens!(st, p, gates_nat, st.sites, p.L)

    st.ni = calculateDensity(st, p) # recalculate densities after changing active space
   
    # exact replacement for st.F = Ub * st.F * G_emp' * G_active' * V_nat'
    evolve_F_left!(st.F, Ub)
    apply_givens_right_dag!(st.F, gates_emp)
    apply_givens_right_dag!(st.F, gates_active)
    apply_givens_right_dag!(st.F, gates_nat)
 
    reduceActive!(st, p, pop)

    return nothing
end

function main(p::SimParams)
    st = initializeSimState(p)

    Hb, eta = generateHamiltonianCoefficients(p)
    # calculate FBR Hamiltonian and bath evolution operator
    etaFBR = eta + 1im * p.dt/2 .* Hb * eta
    Ub = exp(-1im * p.dt * Hb)


    for X in 1:p.nsteps
        st.step += 1
        st.t += p.dt
        @show st.t, st.step

        step!(st, p, Ub, etaFBR)
        measure!(st, p)


        #=
            plot(st.times, st.sigma_x_expectation, xlabel="Time", ylabel="⟨σₓ⟩", title="Qubit Dynamics", legend=true)
            println("Plotting with TEMPO! Check data is right!!")
            df = CSV.read("/Users/lt21314/Documents/PhD/Code/Working Code/Few Body Algorithms/sx_alpha_0.1__omegac_5.0_dt_0.03125_tend_5.0_tcut_5.0_epsrel_2e-07_expohmic_finiteT.csv", DataFrame)
            plot!(df.t, df.sx, linestyle=:dash, label="TEMPO")
            savefig("impurity_sx_expectation_SBM.pdf")
        =#
        results_file = "SubOhmic/SMB_QN_LBO_DynamicalMap_alpha=$(p.alpha)_wc=$(p.wc)_beta=$(p.beta)_s=$(p.s)_epsilonsigmaz=$(p.epsilon_sigmaz)_delta=$(p.delta)_epsilon=$(p.epsilon)_L=$(p.L)_dt=$(p.dt)_probBosonDim=$(p.probabilityBosonDim)_bathtype=exponential.jld2"

        if p.save_results
            jldopen(results_file, "w") do f
                f["times"] = st.times
                f["dynamicalmaps"] = st.dynamicalmaps
                f["propogators"] = st.propogators
            end
        end

    end
    return nothing
end


alpha = parse(Float64, ARGS[1])
wc = 5.0
beta = 10000000
s = 0.3
epsilon_sigmaz = 0.0

T = 250 # end time
dt = 0.025

steps = Int(T/dt)

p = SimParams(
    2500, # L
    dt, # dt
    steps, # nsteps
    wc, # wc, exponential cutoff freq for exp ohmic bath
    alpha, # alpha, coupling strength
    epsilon_sigmaz, # epsilon_sigmaz, impurity energy splitting
    1.0, # delta, sigma_x term strength
    beta, # beta, inverse temperature for finite T runs
    s, # s
    1E-7, # probability needed of max occupation to increase local dim
    1E-8, # epsilon, tolerance for empty/full
    1E-8, # apply_cutoff, cutoff for applying truncation after gates
    1E-8, # time_evolution_cutoff, cutoff for truncation during time evolution
    40, # initDim, initial local dimension for bath modes 1 and 2, which are needed to be large to capture initial state with ancilla
    false, # save_results, whether to save results to file
)

main(p)
