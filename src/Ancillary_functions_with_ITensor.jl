module Ancillary_functions
    using LinearAlgebra
    using Kronecker
    using Plots
    using ITensors

    export noise_removal
    export fidelity
    export replace_nan
    export trace_dist
    export Id_check
    export ρ_test
    export get_inv
    export map_to_principal
    export matrix_log
    export interleaved_to_separated
    export create_Λ
    export lmult
    export rmult
    export vectorise_mat
    export compute_dΛdt
    export compute_overlap
    export current_plots_NESS
    export op_mpo

    
    
    """
    Test functions
    """

    function noise_removal(ρ;kwargs...)
        """
        This function removes any numerical noise that can interfere with the diagonalisation. 
        """
        cutoff = get(kwargs,:noise_cutoff,1e-15)
        real_ρ = real.(ρ)
        imag_ρ = imag.(ρ)
        imag_ρ[abs.(imag_ρ) .< cutoff] .= 0
        real_ρ[abs.(real_ρ) .< cutoff] .= 0
        ρ = real_ρ +im*imag_ρ
    return ρ
    end
    
    function fidelity(ρ,σ)
        """
        Computes the fidelity of two density matrices ρ and σ.
        """
        matrix = sqrt(ρ)*σ*sqrt(ρ)
        matrix = sqrt(matrix)
        fid = tr(matrix)*conj(tr(matrix))
        return fid
    end

    function replace_nan(J)
        if typeof(J) <: AbstractVector
            replace!(J, NaN => J[2])
        elseif isnan(J)
            J = 0
        end
        return J
    end

    function trace_dist(ρ,σ)
        """
        Computes the trace distance of two density matrices ρ and σ.
        """
        matrix = (ρ-σ)*(ρ-σ)'
        matrix = sqrt(matrix)
        return 0.5*tr(matrix)
    end

    function Id_check(Λmat)
        """
        Returns the size of the difference between Λmat and the identity matrix.
        """
        n = size(Λmat)[1]
        Id = zeros(n,n)
        for i=1:n
            Id[i,i] = 1
        end
        return opnorm(Λmat-Id)
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

    function get_inv(Λτ)
        """
        Inverts Λτ, using the pseudo inverse if needed.
        """
        try
            Λτ_inv = inv(Λτ)
        catch
            Λτ_inv = pinv(Λτ)
        end
        Λτ_pinv = pinv(Λτ)
        if Id_check(Λτ*Λτ_pinv)<Id_check(Λτ*Λτ_inv)
            Λτ_inv = Λτ_pinv
        end
        return Λτ_inv
    end

    function map_to_principal(z)
        """
        Maps z to its principle value in the complex plain.
        """
        im_ = imag(z)
        im_ = im_ - 2*π*floor((im_ + π)/(2*π))
        return Complex(real(z), im_)
    end

    function matrix_log(A)
        """
        Takes the matrix logarithm using spectral decomposition.
        """
        F = eigen(A)
        D = Diagonal(F.values)
        V = F.vectors
        V_inv = inv(V)
        #err = A .- V*D*V_inv
        
        principal_log_D = map_to_principal.(log(D))
        log_A = V*principal_log_D*V_inv
        
        return log_A
    end


    function interleaved_to_separated(G)
        """
        Changes the mode ordering of the matrix G from interleaved to separated.
        """
        n = Int((size(G)[1])/2)
        
        G_sep = similar(G)
        for i =1:n
            for j=1:n
                G_sep[i,j] = G[2*i-1,2*j-1]
                G_sep[i+n,j] = G[2*i,2*j-1]
                G_sep[i,j+n] = G[2*i-1,2*j]
                G_sep[i+n,j+n] = G[2*i,2*j]
            end
        end
        return G_sep
    end
    

    function create_Λ(L_list,dt)
        """
        Takes a list of time local generators and computes the corresponding map for each.
        """
        L_exp = exp.(L_list*dt)
        Λ = reduce(*,reverse(L_exp))
        return Λ
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

    function compute_dΛdt(Λs,δt)
        """
        Computes derivative using simple difference formula.
        """
        dΛτ = Λs[3] - Λs[1]
        dΛτdt = dΛτ/(2*δt)
        return dΛτdt
    end

    function compute_overlap(A,B;kwargs...)
        overlap_choice = get(kwargs,:overlap_choice,"trace")
        if overlap_choice == "trace"
            A_mat = unvectorise_ρ(A,false)
            B_mat = unvectorise_ρ(B,false)
            overlap = tr(A_mat*B_mat)
        elseif overlap_choice == "inner"
            overlap = sum(A .*B)
        end
        return overlap
    end
    

    """
    The functions below are not used in any methods currently.
    """

        
    function current_plots_NESS(JL_list,JR_list,den_list,Jp,times,NESS_bool,P)
        (;D) = P
        if NESS_bool
            current_title = "NESS currents"
            density_title = "NESS density"
        else
            current_title = "Currents"
            density_title = "Density"
        end
        Jp = Jp*ones(length(times))
        fac = 100/D
        plot(times,real.(fac*JL_list),label="JL",xlabel="Time",ylabel="Particle current")
        plot!(times,real.(fac*JR_list),label="JR")
        #plot(times,real.(Jp))
        display(plot!(times,real.(fac*Jp),label="LB result",title =current_title))
        display(plot(times,real.(den_list),label="Density",title = density_title))
    end

    function U_thermo(N,f_k)
        """
        This unitary maps the energy eigenmode basis to the thermofield basis.
        """
        U_th = zeros(N,N)
        U_th[1,1],U_th[2,2] = 1,1
        b = 0
        for i=3:2:N
            b += 1
            U_th[i,i],U_th[i+1,i+1] = sqrt(1-f_k[b]),-sqrt(1-f_k[b])
            U_th[i,i+1],U_th[i+1,i] = sqrt(f_k[b]),sqrt(f_k[b])
        end
        return U_th
    end

    function U_chain(N,U1,U2)
        """
        This unitary maps the thermofield basis to the tridiagonal thermofield basis. 
        """
        U_tot = zeros(N,N)
        U_tot[1,1],U_tot[2,2] = 1,1
        b1 = 0
        for i=3:2:N
            b2 = 0 # resets the column iteration
            b1 +=1
            for j =3:2:N
                b2 += 1

                U_tot[i,j] = U1[b1,b2]
                U_tot[i+1,j+1] = U2[b1,b2]
            end
        end
        return U_tot
    end

    function chain_to_star(A)
        """
        This function takes a tridiagonal matrix (chain geometry) and maps it to a star geometry. 
        Maybe just make it a function to extract just the unitary, and put a check in for whether its 
        a star?
        """

        N = length(A[1,:])
        U = zeros(N,N)
        A_sub = A[2:N,2:n]
        U_sub = eigen(A_sub).vectors
        U[1,1] = 1
        U[2:N,2:N] = U_sub
        A_star = U'*A*U
        return A_star
    end

    function op_mpo(sites, which_op, j)
        left_ops = "Id"
        right_ops = "Id"
        if has_fermion_string(which_op, sites[j])
          left_ops = "F"
        end
        ops = [n < j ? left_ops : (n > j ? right_ops : which_op) for n in 1:length(sites)]
        return MPO([op(ops[n], sites[n]) for n in 1:length(sites)])
      end

end
