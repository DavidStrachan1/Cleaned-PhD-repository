module map_functions

    using ITensors
    using ITensorMPS
    using PolyChaos
    using LinearAlgebra
    using Plots
    using Observers #for TDVP
    using Kronecker 


    export calculateDynamicalMap
    export ancilla_phase_gate_swap
    export system_swaps
    export swap_gate
    export rdm_para
    export apply_gates_to_ρ
    export choi_isomorphism
    export create_Choi_state
    export unprime_ind
    export QN_matching
    export QNumbers
    export removeqns_mode

    # #NESS_fn and associated stuff
    function calculateDynamicalMap(ψ_input,layout,ordering_choice,symmetry_subspace;kwargs...)#ψ_input,P,DP,ordering_choice;kwargs...)
        
        use_spin_operators = get(kwargs,:use_spin_operators,false)
        Ns = length(layout.system)
        s = siteinds(ψ_input)

        cdag = [op(s,"Cdag",n) for n in 1:length(ψ_input)]
        c = [op(s,"C",n) for n in 1:length(ψ_input)]
        
        ψ = deepcopy(ψ_input)
        if !use_spin_operators && ordering_choice == "interleaved"
            ##swap phases.
            gates =  ancilla_phase_gate_swap(layout.ancilla,Ns,c,cdag)
            ψ = apply_gates_to_ψ(ψ,gates)
        end

        if ordering_choice == "interleaved"
            ###Applies fermionic swap gates to change the order from interleaved to separated.
            ψ = system_swaps(ψ,layout.system[1],Ns,c,cdag)
        end
        ##calculate reduced density matrix.
        rm_inds =  [layout.system;layout.ancilla]
        ρ = rdm_para(ψ_input,rm_inds,layout,symmetry_subspace)

        ###applies a particle hole transformation to the ancilla states.
        gates = [cdag[n] + c[n] for n in layout.ancilla]
        ρ = apply_gates_to_ρ(ρ,gates,true)

        ###Convert Tensor to matrix.
        cutoff = 1e-5
        ρmat,Λmat = choi_isomorphism(ρ,layout,s)
        return Λmat
    end
    function ancilla_phase_gate_swap(inds,Ns,c,cdag)

        Uph_list = []
        for i =1:Ns
            for j =1:(i-1)
                x = prime(cdag[ind[j]])*c[ind[j]]*prime(c[ind[i]])*cdag[ind[i]]
                unprime_ind(inds(x)[1],x)
                unprime_ind(inds(x)[3],x)  
                Rinds = inds(x,plev=0)
                Linds = Rinds'
                Uph = exp(-im*π*x,Linds,Rinds)
                push!(Uph_list,Uph)
            end
        end
        return Uph_list
    end
    function system_swaps(ψ_input,start,Ns,c,cdag;kwargs...)#,DP,P;kwargs...)

        """
        This function takes a state where the system and ancilla modes are interleaved
        (with the first system mode at the start site) and swaps the ordering such that 
        indices start:start+Ns-1 are all the system modes and start+Ns:start+2Ns-1 are
        the ancilla modes.
        The index start+2(i-1) gives the site index of the ith system mode.

        """


        ψ = deepcopy(ψ_input)
        use_spin_operators = get(kwargs,:use_spin_operators,false)
        s = siteinds(ψ)

        for i=2:Ns
            for j = 1:(i-1)
                ind = start+2(i-1)-j
                ind1 = ind
                ind2 = ind+1
                swap = swap_gate(ind1,ind2,c,cdag,s;use_spin_operators)  
                orthogonalize!(ψ,ind)
                wf = (ψ[ind] * ψ[ind+1]) * swap
                noprime!(wf)
                inds3 = uniqueinds(ψ[ind],ψ[ind+1])
                U,S,V = svd(wf,inds3,cutoff=0)
                ψ[ind] = U
                ψ[ind+1] = S*V
            end
        end
        return ψ
    end
    function swap_gate(i,j,c,cdag,s;kwargs...)

        use_spin_operators = get(kwargs,:use_spin_operators,false)
        
        swap =  cdag[i]*c[j]+cdag[j]*c[i]
        N1 = prime(cdag[i])*c[i]
        N2 = prime(cdag[j])*c[j]
        N1_dag = prime(c[i])*cdag[i]
        N2_dag = prime(c[j])*cdag[j]

        unprime_ind(inds(N1)[1],N1)
        unprime_ind(inds(N2)[1],N2)
        unprime_ind(inds(N1_dag)[1],N1_dag)
        unprime_ind(inds(N2_dag)[1],N2_dag)  

        N = N1*N2  
        N_dag = N1_dag*N2_dag
        swap = swap + N + N_dag

        if !use_spin_operators
            Rinds = inds(N,plev=0)
            Linds = Rinds'
            fermi_fac=  exp(-im*π*N,Linds,Rinds)
            swap = swap*prime(fermi_fac)
            unprime_ind(inds(swap)[3],swap)
            unprime_ind(inds(swap)[4],swap)
        end
        return swap
    end
    function rdm_para(ψ_input,rm_inds,layout,symmetry_subspace)#DP,P)
        # (;Ns,N_L,N_R,symmetry_subspace) = P
        # (;q,s,N,qtot) = DP

        Ns = length(layout.system)
        N_L = 2*length(layout.left_filled)
        N_R = 2*length(layout.right_filled)
        s = siteinds(ψ_input)
        N = length(s)
        qtot = 1:length(s)

        N_inds = length(rm_inds)
        left_bath_bool,right_bath_bool = N_L>0,N_R>0
        d = NDTensors.dim(s[1])

        ψ = deepcopy(ψ_input)
        ψdag = dag(ψ)
        ITensors.prime!(linkinds, ψdag)
        rdm_ = ITensor(dag(s[rm_inds]),s[rm_inds]')
        ρl =  1
        ρr =  ψdag[N]*ψ[N]

        ##Trace out left bath
        if left_bath_bool
            ρl =  ψdag[1]*ψ[1]
            left_inds = qtot[2:rm_inds[1]-1]
            for k in left_inds
                ρl = ρl* ψdag[k]
                ρl = ρl* ψ[k]
            end
        end


        ##Trace out right bath
        if right_bath_bool
            right_inds = qtot[rm_inds[end]+1:end-1]
            for k in reverse(right_inds)   
                ρr = ρr* ψdag[k]
                ρr = ρr* ψ[k]
            end
        end


        for i=0:(d^(2*N_inds) - 1)  
            #The first 2Ns (N_inds) indices are taken as s[rm_inds] and the last 2Ns (N_inds) indices are taken as s[rm_inds]'.
            ##I'm deliberately contracting ψdag[j] and ψ[j] with ρ separately to prevent creating a tensor of size
            ##χ^4 with χ being the local bond dimension. 
            ##The largest tensor created is of size \chi^2*d where d is the site dimension (2).
            v = Vector{Any}(undef,N_inds)
            w = Vector{Any}(undef,N_inds)
            
            #creating a ditstring (bitstring of base d), representing the indices of the density matrix.
            #x gives the indices of s[q], y gives the indices of s[q]'.
            ditstring = zeros(Int,2*N_inds)
            dit = reverse(digits(i,base=d))
            ditstring[(end-(length(dit))+1):end] = dit
            x = ditstring[1:N_inds] .+1
            y = ditstring[N_inds+1:end] .+1

            if QN_matching(x,y,d,symmetry_subspace)
                local ρ = copy(ρl)
                b = 0
                for k in rm_inds
                    b += 1
                    C1 = ψ[k]*onehot(dag(s[k])=>x[b])
                    C2 = ψdag[k]*onehot(s[k]=>y[b])
                    ρ = ρ*C1
                    ρ = ρ*C2
                    v[b] = dag(s[k]) => x[b]
                    w[b] = s[k]' => y[b]
                end
                ρ = ρ*ρr
                rdm_[v...,w...] = ρ[1]
            end
        end
        return rdm_
    end
    function apply_gates_to_ρ(ρ,gates,truncate_bool)

        vector_gates_bool = isa(gates, Vector)
        vector_ρ_bool =  isa(ρ, Vector)

        if vector_gates_bool+vector_ρ_bool == 2
            println("functionality of apply_gates_to_ρ not implemented")
        end
        if vector_ρ_bool == true
            ρ_copy = gates
            gates_copy = ρ
            ρ = ρ_copy
            gates = gates_copy
        end
            
        if isa(gates, Vector)
            for i=1:length(gates)    
                if typeof(gates[i]) == MPO
                    if truncate_bool
                        ρ = apply(gates[i],ρ; cutoff = 1e-15)
                        ρ = apply(ρ,gates[i]; cutoff = 1e-15)
                    else
                        ρ = apply(gates[i],ρ;alg="naive",truncate=false)
                        ρ = apply(ρ,gates[i];alg="naive",truncate=false)
                    end
                else
                    ρ = apply(gates[i],ρ)
                    ρ = apply(ρ,gates[i])
                end
            end
        elseif typeof(gates) == MPO
            if truncate_bool
                ρ = apply(gates,ρ; cutoff = 1e-15)
                ρ = apply(ρ,gates; cutoff = 1e-15)
            else
                ρ = apply(gates,ρ;alg="naive",truncate=false)
                ρ = apply(ρ,gates;alg="naive",truncate=false)
            end
        else
            ρ = apply(gates,ρ)
            ρ = apply(ρ,gates)
        end
        return ρ


    end
    function choi_isomorphism(ρ,layout,s)

        Ns = length(layout.system)

        d = NDTensors.dim(s[1])^Ns
        s = removeqns_mod(s)
        ρ = removeqns_mod(ρ)
        Cs = combiner(reverse(s[layout.system])) # Combiner tensor for merging system legs into a fat index
        Ca = combiner(reverse(s[layout.ancilla])) # Combiner tensor for merging ancilla legs into a fat index
        ρΛ = ρ*dag(Cs)*Cs'*dag(Ca)*Ca'# Merge physical legs to form a density matrix

        Css = combiner([inds(Cs)[1]',dag(inds(Cs)[1])])
        Caa = combiner([inds(Ca)[1]',dag(inds(Ca)[1])])
        Csa = combiner([inds(Cs)[1],inds(Ca)[1]])
        ρmat = ρΛ*dag(Csa)*Csa'
        ρmat = Matrix(ρmat,inds(ρmat));
        
        Λmat = d*ρΛ*Css*Caa
        Λmat = conj(Matrix(Λmat,inds(Λmat)));
        return ρmat,Λmat
    end
    function create_Choi_state(ψ_input,layout)#P,DP;kwargs...)
        
        """
        This function is a way of getting round the weird behaviour of the Jordan Wigner strings
        inside ITensor (https://itensor.discourse.group/t/are-jordan-wigner-strings-handled-in-apply/2266).
        This initialises the Choi state in separated form with no Jordan Wigner operators considered, by initialising in 
        interleaved format which doesn't implement any JW factors, then applying qubit swap gates.
        """

        """
        NOTE:interleaved_inds to reverse(interleaved_inds) was changed in January 2026.
        """

        s = siteinds(ψ_input)
        cdag = [op(s,"Cdag",n) for n in 1:length(s)]
        c = [op(s,"C",n) for n in 1:length(s)]
        q = [layout.system;layout.ancilla]
        Id = Vector{ITensor}(undef,length(s))                    #List of MPS identities
        for i =1:length(s)
            iv = s[i]
            ID = ITensor(iv', dag(iv));
            for j in 1:ITensors.dim(iv)
                ID[iv' => j, iv => j] = 1.0
            end
            Id[i] = ID
        end

        ψ = deepcopy(ψ_input)
        interleaved_inds = q[1:2:end]
        system_gate = [(cdag[n]*Id[n+1] + Id[n]*cdag[n+1])/sqrt(2) for n in reverse(interleaved_inds)]
        ψ = apply(system_gate,ψ;cutoff=1e-15)
        ψ = system_swaps(ψ,q[1],length(layout.system),c,cdag;use_spin_operators = true)
    
        return ψ 
    end
    function unprime_ind(ind,x)
        int = 1
        j = setprime(ind,int)
        replaceind!(x,ind,j)
        return x
    end
    function QN_matching(x,y,d)
        
        QN_x = sum(x .- 1)#QNumbers(x,d)
        QN_y = sum(y .- 1)
        if QN_x == QN_y
            return true
        else
            return false
        end
    
    end
    
    function removeqns_mod(x)
        try 
            x = removeqns(x)
        catch
            if typeof(x) == MPO
                x = MPO([removeqns(x[i]) for i =1:length(x)])
            elseif typeof(x) == MPS
                x = MPS([removeqns(x[i]) for i =1:length(x)])
            end
        end
        return x
    end

end