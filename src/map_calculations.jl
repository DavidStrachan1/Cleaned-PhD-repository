module map_calculations

    using ITensors
    using ITensorMPS
    using PolyChaos
    using LinearAlgebra
    using Kronecker
    using SparseArrays
    using ProgressMeter


    export propagate_correlations
    export spin_operators
    export matrix_operators
    export JW_string_mat
    export calculate_ρ_from_correlation_matrix
    export calculate_Λ_from_correlation_matrix
    export ρ_to_Λ
    export extract_physical_modes
    export N_superoperator
    export ρ_test
    export expand_Λ
    export vectorise_mat
    export compute_maps
    export compute_spectra
    export differentiate
    export unvectorise_ρ
    export map_to_principal
    export NESS_extraction
    export lmult
    export rmult

    export fermionic_spin_ring_lindblad
    export markovian_evolution
    export number_operator

    function propagate_correlations(Ci,H_single,times)
        """
        The correlation matrix C_ij = expect(cdag[j]*c[i]) propagates according to
        C_ij(t) =U*C_ij(0)*U', but G_ij = expect(cdag[i]*c[j]) doesn't.This is why 
        the correlation matrices are transposed before calculating the reduced density
        matrix as the formula uses the second definition.
        """
        δt = times[2] - times[1]
        
    
        U_step = exp(-im*δt*H_single)

        corrs = Vector{Any}(undef,length(times))
        corrs[1] = U_step*Ci*U_step'
        for i in 2:length(times)
            corrs[i] = U_step*corrs[i-1]*U_step'
        end
        return corrs
    end
    function spin_operators(M)

        # Build sparse matrix version of basic spin (Pauli) operators :
        sp = spdiagm(2,2,1=>ones(1))
        sm = spdiagm(2,2,-1=>ones(1))
        sz = spdiagm(2,2,0=>[1;-1]);
        num = spdiagm(2,2,0=>[0;1])
        # Notice there are NO factors of (1/2) for spin-1/2 included here.

        # Construct spin operators for each spin in the full Hilbert space :
        Sz = Vector{Any}(undef, M)
        Sp = Vector{Any}(undef, M)
        Sm = Vector{Any}(undef, M)
        Num = Vector{Any}(undef,M)
        for m=1:M
            Sz[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),sz),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
            Sp[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),sp),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
            Sm[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),sm),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
            Num[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),num),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
        end
        return Sz,Sp,Sm,Num
    end
    function matrix_operators(M)

        Sz,Sp,Sm,_ = spin_operators(M)
        cdag_mat = Vector{Any}(undef,M)
        c_mat = Vector{Any}(undef,M)

        for n=1:M
            #Build JW_string
            Z = JW_string_mat(Sz,n,M)
            cdag_mat[n] = Z*Sm[n]
            c_mat[n]  = Z*Sp[n]
        end
        return cdag_mat,c_mat
    
    end
    function JW_string_mat(Sz,site,M;kwargs...)
        inds = get(kwargs,:inds,1:(site-1))

        Z = 1.0*Matrix(I, 2^M, 2^M)
        for i in inds
            Z = Z*Sz[i];
        end
        return Z
    end
    function calculate_ρ_from_correlation_matrix(C, mode_subset; eps=1e-14,kwargs...)
        """
            gaussian_rdm_density_matrix(C, mode_subset,qA; eps=1e-14)

        Construct the fermionic Gaussian density matrix

            ρ = ⊗ₖ [(1-νₖ)|0><0| + νₖ|1><1|]

        from the correlation matrix C restricted to the mode_subset modes.

        Arguments
        ---------
        C            : correlation matrix
        mode_subset  : modes to calculate rdm
        qA           : ancilla modes

        Returns
        -------
        ρ      : Dense many-body density matrix
        """

        N = length(mode_subset)
        ddag,d = matrix_operators(N)
        dim = size(ddag[1],1)

        #take the subset and symmetrize
        C = transpose(C[mode_subset,mode_subset])
        C = (C + C') / 2

        # 2. Diagonalize correlation matrix
        eig = eigen(Hermitian(C))
        ν = clamp.(real(eig.values), eps, 1-eps)
        U = eig.vectors

        # Construct rotated fermion operators
        #    f_k = Σ_i U†_{ki} d_i
        f = Vector{Any}(undef, N)
        fdag = Vector{Any}(undef, N)

        for k in 1:N
            fk = zero(d[1])
            for i in 1:N
                fk += conj(U[i,k]) * d[i]
            end
            f[k] = fk
            fdag[k] = fk'
        end

        # Construct many-body density matrix
        #    ρ = Π_k [(1-ν_k)(1-n_k) + ν_k n_k]
        # where n_k = f†_k f_k

        ρ = Matrix(I, dim, dim)
        for k in 1:N
            nk = fdag[k] * f[k]
            ρk = (1-ν[k]) * (Matrix(I, dim, dim) - nk) +
                ν[k] * nk
            ρ *= ρk
        end

        ##checks it's a valid density matrix up to a tolerance 1e-5
        ρ_test(ρ,1e-5) 

        return ρ
    end
    function calculate_Λ_from_correlation_matrix(C, mode_subset,qA; eps=1e-14,kwargs...)
        """
            gaussian_rdm_density_matrix(C, mode_subset,qA; eps=1e-14)

        Construct the fermionic Gaussian density matrix

            ρ = ⊗ₖ [(1-νₖ)|0><0| + νₖ|1><1|]

        from the correlation matrix C restricted to the mode_subset modes.

        Arguments
        ---------
        C            : correlation matrix
        mode_subset  : modes to calculate rdm
        qA           : ancilla modes

        Returns
        -------
        ρ      : Dense many-body density matrix
        """

        symmetry_subspace = get(kwargs,:symmetry_subspace, "Number conserving")
        N = length(mode_subset)
        N_ring = length(qA)
        ddag,d = matrix_operators(N)
        dim = size(ddag[1],1)

        #take the subset and symmetrize
        C = transpose(C[mode_subset,mode_subset])
        C = (C + C') / 2


        # 2. Diagonalize correlation matrix
        eig = eigen(Hermitian(C))
        ν = clamp.(real(eig.values), eps, 1-eps)
        U = eig.vectors

        # Construct rotated fermion operators
        #    f_k = Σ_i U†_{ki} d_i
        f = Vector{Any}(undef, N)
        fdag = Vector{Any}(undef, N)

        for k in 1:N
            fk = zero(d[1])
            for i in 1:N
                fk += conj(U[i,k]) * d[i]
            end
            f[k] = fk
            fdag[k] = fk'
        end

        # Construct many-body density matrix
        #    ρ = Π_k [(1-ν_k)(1-n_k) + ν_k n_k]
        # where n_k = f†_k f_k

        ρ = Matrix(I, dim, dim)
        for k in 1:N
            nk = fdag[k] * f[k]
            ρk = (1-ν[k]) * (Matrix(I, dim, dim) - nk) +
                ν[k] * nk

            ρ *= ρk
        end

        # gates =  ancilla_phase_gate_swap(qA,ddag,d)
        # ρ = apply_gates_to_ρmat(ρ,gates)

        ##PH transform
        for index in qA
        #      gate = Sp[index] + Sm[index]
            gate = ddag[index] +d[index]#+ ((-1)^(N_ring))*c_mat[index]
            ρ = gate*ρ*gate'
        end

        ##checks it's a valid density matrix up to a tolerance 1e-5
        ρ_test(ρ,1e-5) 

        ##reshape to give the map
        Λ = ρ_to_Λ(ρ,N_ring) 
        if symmetry_subspace =="Number conserving"
            qN = extract_physical_modes(N_ring)
            Λ = Λ[qN,qN]
        end

        return Λ
    end
    function compute_maps(corrs,δt,sys_anc_inds,ancilla_inds;kwargs...)
        Λ_vec = similar(corrs)
        L_vec = Vector{Any}(undef,length(Λ_vec)-2)
        println("Calculating Λ(t) from the single particle correlation matrix.")
        @showprogress for i =1:length(corrs)
            Λ_vec[i] = map_calculations.calculate_Λ_from_correlation_matrix(corrs[i],sys_anc_inds,ancilla_inds; eps=1e-14,kwargs...)
            if i>2
                L = (Λ_vec[i] - Λ_vec[i-2])/(2*δt)
                L = L * pinv(Λ_vec[i-1])
                L_vec[i-2] = L
            end
        end
        return Λ_vec,L_vec
    end
    function compute_spectra(Λ_vec,L_vec)
        spectra_Λ = complex(zeros(length(Λ_vec),size(Λ_vec[1])[1]))
        spectra_L = complex(zeros(length(L_vec),size(Λ_vec[1])[1]))
        println("computing map spectra")
        [spectra_Λ[i,:] = eigen(Λ_vec[i]).values for i=1:length(Λ_vec)] 
        println("computing propagator spectra")
        [spectra_L[i,:] = eigen(L_vec[i]).values for i=1:length(L_vec)] 
        return spectra_Λ,spectra_L
    end
    function NESS_extraction(map,Λ_or_L,Nsys;kwargs...)

        symmetry_subspace = get(kwargs,:symmetry_subspace, "Number conserving")
        spec = eigen(map).values
        vecs = eigen(map).vectors
        spec = map_to_principal.(spec)

        vec = zeros(ComplexF64,2^(2*Nsys))
        qN = map_calculations.extract_physical_modes(Nsys)

        if Λ_or_L == "Λ"
            ind = argmax(real.(spec))
        elseif Λ_or_L == "L"
            ind = argmin(abs.(spec))
        end
        if symmetry_subspace == "Number conserving"
            vec[qN] = vecs[:,ind]
        else
            vec = vecs[:,ind]
        end
        vec =unvectorise_ρ(vec,true)
        return vec
    end

    function ρ_to_Λ(ρ,Ns)
        d = 2^Ns
        Λ = zeros(ComplexF64, d^2, d^2) 
        for i_s=1:d, j_s=1:d
            for i_a=1:d, j_a=1:d  
                Λ[(i_s-1)*d + j_s, (i_a-1)*d + j_a] = 
                    conj(d*ρ[(i_s-1)*d + i_a, (j_s-1)*d + j_a])
            end
        end
        return Λ
    end
    function extract_physical_modes(Ns)
        diag_vals = diag(N_superoperator(Ns))
        qN = findall(x->x==0,diag_vals)
        return qN
    end
    function N_superoperator(Ns)
        d = 2^Ns
        Id = Diagonal(ones(Float64,d))
        Num = spin_operators(Ns)[4]
        NS = kronecker(Id,sum(Num)) - kronecker(sum(Num),Id)
        return NS
    end
    function ρ_test(ρ,cutoff)
        """
        Tests if ρ is a density matrix up to numerical error "cutoff"/
        """
        x = eigen(ρ).values
        bool = false
        message0 = "Valid density matrix up to"*string(cutoff)
        message = "Not a density matrix:"
        if minimum(real.(x))<-cutoff
            bool = true
            message *= "minimum of spectrum<"*string(cutoff)*","
        end
        if maximum(real.(x))-1>cutoff
            message *= "maximum of spectrum>1,"
            bool = true
        end
        if maximum(imag.(x))>cutoff
            message *= "max(imag(spectrum))>"*string(cutoff)*","
            bool = true
        end
        if  abs(1-sum(x))>cutoff
            message *= "1-tr(ρ) >"*string(cutoff)*","
            bool = true
        end
        if norm(ρ-ρ')>cutoff
            message *= "ρ-dag(ρ)>"*string(cutoff)*","
            bool = true
        end
        if bool
            return message,bool
        else
            return message0,bool
        end
    end
    function expand_Λ(Λ,Ns)
        qN = extract_physical_modes(Ns)
        Nλ = 2^(2*Ns)
        A = complex(zeros(Nλ,Nλ))
        A[qN,qN] = Λ
        return A
    end
    function expand_ρ(ρ,Ns)
        qN = extract_physical_modes(Ns)
        Nρ = 2^(2*Ns)
        A = zeros(ComplexF64,Nρ)
        A[qN] = ρ
        return A 
    end
    function vectorise_mat(mat)
        "Takes a matrix and vectorises it according to the Choi-Jamiolkowski ispmorphism."
        d =  size(mat)[1]
        vec = complex(zeros(Int(d*d)))
        for i =1:d
            for j=1:d
                vec[Int((i-1)*d +j)] = mat[j,i]
            end
        end
        return vec
    end
    function differentiate(x,δt)
        return [(x[i+2] - x[i])/(2*δt) for i in 1:length(x)-2]
    end    
    function unvectorise_ρ(ρvec,tr_bool)

        d =  Int(sqrt(length(ρvec)))
        ρ = complex(zeros(d,d))
        for i =1:d
            for j=1:d
                ρ[j,i] = ρvec[Int((i-1)*d +j)]
            end
        end
        if tr_bool
            ρ = ρ/tr(ρ) ##ensures correct normalisation
        end
        return ρ
    end
    function map_to_principal(z)
        """
        Maps z to its principle value in the complex plain.
        """
        im_ = imag(z)
        im_ = im_ - 2*π*floor((im_ + π)/(2*π))
        return Complex(real(z), im_)
    end

    function rmult(A)
        """
        Super-operator representing right-multiplication on a vectorised density matrix
        """
        d = size(A)[1]
        Id = 1.0*Matrix(I, d, d)
        return kronecker(transpose(A),Id)
    end
    function lmult(A)
        """
        Super-operator representing left-multiplication on a vectorised density matrix
        """
        d = size(A)[1]
        Id = 1.0*Matrix(I, d, d)
        return kronecker(Id,A)
    end


    """
    Markovian functions
    """
    function fermionic_spin_ring_lindblad(bath,system,dissipation_ind)

        d = 2^system.N_ring
        Id = 1* Matrix(I, d, d)
        Left_vac = vectorise_mat(Id)
        cdag,c = matrix_operators(system.N_ring)

        Hsys = complex(zeros(d,d))

        for i =1:system.N_ring
            Hsys += 2*system.B*cdag[i]*c[i]
            if i <system.N_ring
                Hsys += 2*system.J*cdag[i]*c[i+1]
                Hsys += 2*system.J*cdag[i+1]*c[i]

            elseif i!= 1
                @assert(i==system.N_ring)
            # Hsys[1,i] = 2*system.J
            # Hsys[i,1] = 2*system.J
                Hsys += 2*system.J*cdag[1]*c[i]
                Hsys += 2*system.J*cdag[i]*c[1]
            end
        end


        L_unitary = -im*(rmult(Hsys)- lmult(Hsys))
        f = 1 / (exp(bath.β*(2*system.B-bath.μ)) + 1)
        Γd = (4/π)*bath.Γ*(1-f)
        Γe = (4/π)*bath.Γ*f

        
        emission = Γd*(rmult(cdag[dissipation_ind])*lmult(c[dissipation_ind]) - (1/2)*rmult(cdag[dissipation_ind]*c[dissipation_ind]) - (1/2)*lmult(cdag[dissipation_ind]*c[dissipation_ind]))
        absorption = Γe*(rmult(c[dissipation_ind])*lmult(cdag[dissipation_ind]) - (1/2)*rmult(c[dissipation_ind]*cdag[dissipation_ind]) - (1/2)*lmult(c[dissipation_ind]*cdag[dissipation_ind]))
        L_markovian =L_unitary + absorption + emission

        return L_markovian
    end
    function markovian_evolution(L,ρi,times)
        return [exp(L*t)*ρi for t in times]
    end
    function number_operator(ind,Nsys)
        N = [0 0;0 1]
        Id = Matrix(I,2,2)

        if ind == 1
            N_manybody = N
        else
            N_manybody = Id
        end

        for i=1:Nsys-1
            if i+1 == ind
                N_manybody = kronecker(N_manybody,N)
            else
                N_manybody = kronecker(N_manybody,Id)
            end
        end
        return N_manybody
    end

end