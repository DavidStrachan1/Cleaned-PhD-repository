
    export sweep_over_param_space
    export sweep_over_param_space_general
    export finding_μ_opt
    export sweep_over_μβ_space
    export p_to_V
    export single_qubit_state_preparation
    export single_qubit_analysis
    export Mpemba_param
    export p_optimal

    export thermalise_state

    export ρ_fast_creation


    export thermal_QME_analysis
    export decay_mode_analysis
    export Non_markovian_QME_analysis
    export Instantaneous_NESS_analysis
    export single_qubit_analysis

    export component_breakdown
    export convergence_analysis_deprecated
    export propagate_state
    export map_to_positive
    export convergence_surface_plot
    export finding_μmax
    export L0_density
    export Effective_L_rates
    export propagate_p
    export quench
    export ρf_to_corr_with_ancilla
    export dpdt


    function thermalise_state(β,Λ_or_L,cutoffs,P)
        trace_cutoff, time_cutoff = cutoffs

        P.β_L,P.β_R = β,β
        DP = DP_initialisation(P)
        BP = BP_initialisation(DP,P)
        (;Ci, H_single) = BP
        (;δt) = P

        corrs = Vector{Any}(undef,3)
        bool = false
        tr_list = Any[]
        Nt = round(Int, time_cutoff/δt)
        U_step = exp(-im*δt*H_single)

        corrs[1] = U_step*Ci*U_step'
        ρ_prev = NESS_extraction(calculate_ρΛ_using_G(corrs[1], DP, P)[1], "Λ", P)[2]

        for i in 1:Nt
            corrs[2] = U_step*corrs[1]*U_step'
            corrs[3] = U_step*corrs[2]*U_step'
            Λs = [calculate_ρΛ_using_G(corr,DP,P)[1] for corr in corrs]

            if i != 0 && Λ_or_L == "L"
                L = compute_Louv(Λs,false,P,DP)[1]
                ρ = NESS_extraction(L, "L", P)[2]
            else
                ρ = NESS_extraction(Λs[2],"Λ",P)[2]
            end

            tr_dist = abs(trace_dist(unvectorise_ρ(ρ,true), unvectorise_ρ(ρ_prev,true)))/δt
            push!(tr_list,tr_dist)            

            if tr_dist < trace_cutoff
                bool = true
                break
            end
            ρ_prev = ρ
            corrs[1] = corrs[2]
        end
        if bool ==false
            println("State hasn't thermalised")
        end
        # times = range(δt,stop=Nt*δt,step=δt)
        # layout = Layout(title="log of tr_dist(ρ[i],ρ[i-1])",xaxis_title="Time")
        # #display(plot(times,log10.(tr_list),layout))
        @show(β)
        println("converged up to error "*string(tr_list[end]))
        return ρ_prev,corrs
    end

    function finding_μ_opt(μ_vec,β_vec,Γ_vec,t,P)
        Nμ,Nβ,NΓ = length(μ_vec),length(β_vec),length(Γ_vec)
        Mpemba_tensor = zeros(Nμ,Nβ,NΓ)
        μ_opt_matrix = zeros(Nβ,NΓ)
        for (i,Γ) in enumerate(Γ_vec)
            Mpemba_tensor[:,:,i] = sweep_over_μβ_space(μ_vec,β_vec,Γ,t,P)
        end
        
        for i =1:Nβ
            for j=1:NΓ
                ind = findmax(Mpemba_tensor[:,i,j])[2]
                μ_opt_matrix[i,j] = μ_vec[ind]
            end
        end
        
        plot_contour(μ_opt_matrix,Γ_vec,β_vec,"Γ","β","μ_opt(β,Γ)")
        plot_surface(transpose(μ_opt_matrix),Γ_vec,β_vec,"Γ","β","μ_opt(β,Γ)")
        
        return μ_opt_matrix,Mpemba_tensor
        
    end


    function  Mpemba_param(params,t,P)
        μ,β,Γ = params

        P.β_L,P.β_R = β,β
        P.mu_L,P.mu_R = μ,μ
        P.Γ_L,P.Γ_R = Γ,Γ
        DP = DP_initialisation(P)
        BP = BP_initialisation(DP,P)
        (;δt) = P
        (;Ci, H_single) = BP
        
        corrs = Vector{Any}(undef,3)
        U_step = exp(-im*δt*H_single)
        Ut = exp(-im*(t-δt)*H_single)
        
        
        corrs[1] = Ut*Ci*Ut'
        corrs[2] = U_step*corrs[1]*U_step'
        corrs[3] = U_step*corrs[2]*U_step'
        
        Λs = [calculate_ρΛ_using_G(corr,DP,P)[1] for corr in corrs]
        
        L= compute_Louv(Λs,false,P,DP)[1]

        # if sign(real(L[1,1])) == 1
        #     L[1,1] = -1*L[1,1]
        # end
        # if sign(real(L[2,1])) == -1
        #     L[2,1] = -1*L[2,1]
        # end
        # if sign(real(L[1,2])) == -1
        #     L[1,2] = -1*L[1,2]
        # end
        # if sign(real(L[2,2])) == 1
        #     L[2,2] = -1*L[2,2]
        # end
        if P.Ns == 1
            p_opt,pth = p_optimal(Λs[2],L,P)
            if real(pth) >1
                pth=1
                p_opt = 1
            elseif real(pth)<0
                pth =0
                p_opt = 0
            end
            if real(p_opt) > 1
                p_opt = pth
            elseif real(p_opt)<0
                p_opt = pth
            end
            gap = real(pth - L0_density(BP))
            return real(p_opt-pth),real(p_opt),real(pth),real(gap)
        else
            NESS = NESS_extraction(L,"L",P)[2]
            NESS_mat = unvectorise_ρ(NESS,true)
            ρf = unvectorise_ρ(pinv(expand_Λ(Λs[2],P.Ns))*NESS,true)
            tr_dist = trace_dist(NESS_mat,ρf)
            return tr_dist,ρf,NESS_mat
        end
    end

    function p_optimal(Λ,L,P)
        NESS = NESS_extraction(L,"L",P)[2]
        pth = NESS[4]
        α,β = Λ[1,1], Λ[1,2]
    
        p_opt = (α-(1-pth))/(α-β)
        if real(p_opt)>1
            p_opt = pth
        elseif real(p_opt)<0
            p_opt = pth
        end
        return p_opt,pth
    end

    
    function p_to_V(V_vec,p_opt_mat,pth_mat)
        """
        Assumes V_vec changes along the second dimension of the matrices.
        """
        
        (N1,N2) = size(p_opt_mat)
        


        V_mat = Matrix(reshape(V_vec,(1,N2)))
        V_mat = repeat(V_mat,inner=(N1,1))
        V_opt_mat = similar(V_mat)
        
        for i = 1:N1
            for j= 1:N2
                diff,ind = findmin(abs.(pth_mat[i,:] .- p_opt_mat[i,j]))
                V_opt_mat[i,j] = V_mat[i,ind]
            end
        end
        V_Mpemba_mat = V_opt_mat .- V_mat
        return V_Mpemba_mat
    end

    function sweep_over_param_space_general(p1_vec,p2_vec,p3,control,t,P)
        N1,N2 = length(p1_vec),length(p2_vec)
        Mpemba_vec = zeros(N1,N2)
        fast_state_vec = Vector{Any}(undef,(N1,N2))
        NESS_state_vec = Vector{Any}(undef,(N1,N2))

    
        params =[0,0,0]
        p1_title,p2_title,title = "","",""
        @showprogress for (i,p1) in enumerate(p1_vec)
            for (j,p2) in enumerate(p2_vec)
                if control == "μ"
                    params = [p3,p1,p2]
                    p1_title = "Inverse_temeperature, β"
                    p2_title = "Coupling strength, Γ"
                    title = L"$p^{*}-p_{\textrm{th}}$,μ="*string(p3)
                elseif control =="β"
                    params = [p1,p3,p2]
                    p1_title = "Chemical potential, μ"
                    p2_title = "Coupling strength, Γ"
                    title = L"$p^{*}-p_{\textrm{th}}$,β="*string(p3)
                elseif control == "Γ"
                    params = [p1,p2,p3]
                    p1_title = "Chemical potential, μ"
                    p2_title = "Inverse temperature, β"
                    title = L"$p^{*}-p_{\textrm{th}}$,Γ="*string(p3)
                end
                Mpemba_vec[i,j],fast_state_vec[i,j],NESS_state_vec[i,j] = Mpemba_param(params,t,P)
            end
        end
    


     return Mpemba_vec,fast_state_vec,NESS_state_vec
    end


    
    function sweep_over_param_space(p1_vec,p2_vec,p3,control,t,P)
        N1,N2 = length(p1_vec),length(p2_vec)
        Mpemba_vec = zeros(N1,N2)
        p_opt_vec = zeros(N1,N2)
        p_th_vec = zeros(N1,N2)
        gap_vec = zeros(N1,N2)
    
        params =[0,0,0]
        p1_title,p2_title,title = "","",""
        @showprogress for (i,p1) in enumerate(p1_vec)
            for (j,p2) in enumerate(p2_vec)
                if control == "μ"
                    params = [p3,p1,p2]
                    p1_title = "Inverse_temeperature, β"
                    p2_title = "Coupling strength, Γ"
                    title = L"$p^{*}-p_{\textrm{th}}$,μ="*string(p3)
                elseif control =="β"
                    params = [p1,p3,p2]
                    p1_title = "Chemical potential, μ"
                    p2_title = "Coupling strength, Γ"
                    title = L"$p^{*}-p_{\textrm{th}}$,β="*string(p3)
                elseif control == "Γ"
                    params = [p1,p2,p3]
                    p1_title = "Chemical potential, μ"
                    p2_title = "Inverse temperature, β"
                    title = L"$p^{*}-p_{\textrm{th}}$,Γ="*string(p3)
                end
                Mpemba_vec[i,j],p_opt_vec[i,j],p_th_vec[i,j],gap_vec[i,j] = Mpemba_param(params,t,P)
            end
        end
    

    β_Mpemba_mat = p_to_V(p2_vec,p_opt_vec,p_th_vec)
    μ_Mpemba_mat = transpose(p_to_V(p1_vec,transpose(p_opt_vec),transpose(p_th_vec)))


    # println("Contour or surface plots? Contour = 1,Surface= 2")
    # plot_choice = parse(Int,readline())
    #if plot_choice ==1
        # plot_contour(β_Mpemba_mat,p2_vec,p1_vec,p2_title,p1_title,L"$β^{*}-β$")
        # plot_contour(μ_Mpemba_mat,p2_vec,p1_vec,p2_title,p1_title,L"$μ^{*}-μ$")
        # plot_contour(p_opt_vec,p2_vec,p1_vec,p2_title,p1_title,L"$p^{*}$")
        # plot_contour(p_th_vec,p2_vec,p1_vec,p2_title,p1_title,L"$p^{th}$")
        plot_contour(Mpemba_vec,p2_vec,p1_vec,p2_title,p1_title,title)
  # elseif plot_choice ==2
        # plot_surface(transpose(β_Mpemba_mat),p2_vec,p1_vec,p2_title,p1_title,L"$β^{*}-β$")
        # plot_surface(transpose(μ_Mpemba_mat),p2_vec,p1_vec,p2_title,p1_title,L"$μ^{*}-μ$")
        # plot_surface(transpose(p_opt_vec),p2_vec,p1_vec,p2_title,p1_title,L"$p^{*}$")
        # plot_surface(transpose(p_th_vec),p2_vec,p1_vec,p2_title,p1_title,L"$p^{th}$")
        # plot_surface(transpose(Mpemba_vec),p2_vec,p1_vec,p2_title,p1_title,title)
   # end   
     return β_Mpemba_mat,μ_Mpemba_mat,p_opt_vec,p_th_vec,gap_vec
    end






    function ρ_fast_creation(Λ_subset,L_subset,t_subset,P;kwargs...)
        (;symmetry_subspace) = P
        take_symmetry_subset = get(kwargs,:take_symmetry_subset,true)
        #Extracts steady state from L
        N = length(Λ_subset)
        fast_states = Vector{Any}(undef,N)
        for i = 1:N
            ρ_NESS = NESS_extraction(L_subset[i],"L",P;kwargs...)[2]
            if symmetry_subspace !="Full" && take_symmetry_subset
                Λ = expand_Λ(Λ_subset[i],P.Ns)
            else
                Λ = Λ_subset[i]
            end
            fast_ρ= unvectorise_ρ(pinv(Λ)*ρ_NESS,true)
            ρ_test(fast_ρ,1e-10)
            fast_states[i] = vectorise_mat(fast_ρ)  
        end
        labels = ["ρf(t="*string(t)*")" for t in t_subset]
        return fast_states,labels
    end


    function single_qubit_state_preparation(p_vec,NESS,p_opt)


        p_NESS = real(NESS[end])
        p_vec = [p_NESS;p_opt;p_vec]
        N_states = length(p_vec)
        
        labels = Vector{Any}(undef,N_states)
        states = [Vector{ComplexF64}(undef,4) for i in 1:N_states]
        
        states[1] = vectorise_mat(NESS)
        labels[1] = "NESS,p="*string(p_NESS)
        
        for (i, state) in enumerate(states[2:end])
            state[1] = 1-p_vec[i+1]
            state[4] = p_vec[i+1]
            labels[i+1] = "p="*string(p_vec[i+1])
            if i+1 == 2
                state[1] = 1-p_opt
                state[4] = p_opt
                labels[i+1] = "p_opt ="*string(real.(p_opt))
            end
        end

        return states,labels,p_vec
    end





    function thermal_QME_analysis(Λ_vec,β_vec,cutoffs,NESS_op,P,DP,BP)
        (;times) = DP
        Λ_or_L = NESS_op[2]
        #NESS = unvectorise_ρ(NESS_extraction(NESS_op[1],Λ_or_L,P)[2],true)

        thermal_states = [thermalise_state(β,Λ_or_L,cutoffs,P)[1] for β in β_vec]
        data,title,_ = convergence_analysis(thermal_states,Λ_vec,NESS_op,P,DP,BP)
        labels = ["β="*string(β) for β in β_vec]
        if P.Ns == 1
            p_eff = [" ,p="*string(round(real.(state[4]),digits=3)) for state in thermal_states]
            labels = labels.*p_eff
        end
        labels = [β_vec,labels,"β"]
    
        plot_data(data,times,title,labels)
        return thermal_states,labels
    end


    function Non_markovian_QME_analysis(Λ_vec,L_vec,ind_subset,NESS_op,P,DP,BP)
        (;times) = DP
    # NESS = unvectorise_ρ(NESS_extraction(NESS_op[1],NESS_op[2],P)[2],true)
        Λ_subset = Λ_vec[ind_subset]
        L_subset = L_vec[ind_subset]
        t_subset = times[ind_subset]

        fast_states,labels = ρ_fast_creation(Λ_subset,L_subset,t_subset,P)
        data,title,components = convergence_analysis(fast_states,Λ_vec,NESS_op,P,DP,BP)
        labels = [t_subset,labels,"State extraction time"]


        plot_data(data,times,title,labels)

        return fast_states,labels
    end
        

    function Instantaneous_NESS_analysis(Λ_vec,ind_subset,map_vec,NESS_op,P,DP,BP)
        (;times) = DP
        Λ_or_L = NESS_op[2]
    #  NESS = unvectorise_ρ(NESS_extraction(NESS_op[1],Λ_or_L,P)[2],true)
        map_subset = map_vec[ind_subset]
        t_subset = times[ind_subset]
        
        NESS_states = [NESS_extraction(map,Λ_or_L,P)[2] for map in map_subset]
        labels = ["using t="*string(t)*" NESS" for t in t_subset]
        data,title,_ = convergence_analysis(NESS_states,Λ_vec,NESS_op,P,DP,BP)

        labels = [t_subset,labels,"State extraction time"]
        plot_data(data,times,title,labels)
        
        return NESS_states,labels
    end

    function decay_mode_analysis(k,Λ_vec,ind_subset,map_vec,NESS_op,P,DP,BP)
        (;times) = DP
        Λ_or_L = NESS_op[2]
        NESS = unvectorise_ρ(NESS_extraction(NESS_op[1],Λ_or_L,P)[2],true)
        map_subset = map_vec[ind_subset]
        t_subset = times[ind_subset]

        decay_states = [kth_mode_extraction(k,map,Λ_or_L,P)[2] for map in map_subset]
        decay_states = [map_to_positive(state,NESS) for state in decay_states]
        
        labels = ["using t="*string(t)*" slowest decay mode" for t in t_subset]
        data,title,_ = convergence_analysis(decay_states,Λ_vec,NESS_op,P,DP,BP)

        labels = [t_subset,labels,"State extraction time"]
        plot_data(data,times,title,labels)

        return decay_states,labels
    end

    function single_qubit_analysis(p_vec,Λ_vec,NESS_op,P,DP,BP)
        (;times) = DP
        NESS = unvectorise_ρ(NESS_extraction(NESS_op[1],NESS_op[2],P)[2],true)
        
        Λ =Λ_vec[end]
        pth = NESS[4]
        
        α,β = Λ[1,1], Λ[1,2]

        p_opt = (α-(1-pth))/(α-β)
        
        states,labels,p_vec = single_qubit_state_preparation(p_vec,NESS,p_opt)
        
        data,title,components = convergence_analysis(states,Λ_vec,NESS_op,P,DP,BP)


    # components = list_to_matrix(components)
    # decay_components = abs.(getindex.(components,2))
        labels =[p_vec,labels,"p"]

        
    #     plot_lines(decay_components,labels[2],times,"Decay mode component")
        plot_data(data,times,title,labels)

    end


    function component_breakdown(ρ,map,Λ_or_L,P)
        (;Ns,symmetry_subspace) = P
        
        ρ = vectorise_mat(ρ)
        spec,NESS,_,right_eigvecs = kth_mode_extraction(1,map,Λ_or_L,P)
        norm = tr(unvectorise_ρ(NESS,false))
        right_eigvecs = right_eigvecs./norm
        
        if symmetry_subspace == "Number conserving"
            B = expand_Λ(right_eigvecs,P.Ns)
            components = pinv(B)*ρ
            qN = extract_physical_modes(Ns)
            return components[qN],spec
        else
            components = pinv(right_eigvecs)*ρ
            return components,spec
        end
    end




    function propagate_state(ρ,Λ_vec,NESS,NESS_op,site,P,DP)
        (;Ns,symmetry_subspace) = P
        
        
        states = Vector{Any}(undef,length(Λ_vec))
        tr_dists,corrs = similar(states),similar(states)
        components = similar(states)
        for (i,Λ) in enumerate(Λ_vec)
            if symmetry_subspace == "Number conserving"
                state = unvectorise_ρ(expand_Λ(Λ,P.Ns)*ρ,true)
            else
                state = unvectorise_ρ(Λ*ρ,true)
            end
            tr_dists[i] = trace_dist(NESS,state)
            corrs[i] = ρ_system_corr(state,Ns)
            states[i] = state
            components[i] = component_breakdown(state,NESS_op[1],NESS_op[2],P)[1]

        end
        JL,JR,den = calculate_currents(corrs,site,P,DP)

        return tr_dists,JL,JR,den,components
    end


    function convergence_analysis_deprecated(init_state_vec,Λ_vec,NESS_op,P,DP,BP)
        (;times) = DP
        (;Ns) = P
        NESS = unvectorise_ρ(NESS_extraction(NESS_op[1], NESS_op[2], P)[2],true)
        

        tr_dists_set = Vector{Any}(undef,length(init_state_vec))
        JL_set,JR_set,den_set = similar(tr_dists_set),similar(tr_dists_set),similar(tr_dists_set)
        components_set = similar(JL_set)

        for (i,state) in enumerate(init_state_vec)
            tr_dists_set[i],JL_set[i],
            JR_set[i],den_set[i],components_set[i] = propagate_state(state,Λ_vec,NESS,NESS_op,P,DP,BP) 
        end
        
        
        corr_NESS = ρ_system_corr(NESS,Ns)
        JL_NESS,JR_NESS,den_NESS = current_operator(corr_NESS,P,DP,BP)

        println("distance to NESS measure: trace=0,JL=1,JR=2,den=3,decay mode component=4")
        choice = parse(Int,readline())
        if choice ==0
            data = list_to_matrix(tr_dists_set)
            title = "trace distance to NESS"
        elseif choice == 1
            data = list_to_matrix([(100*(JL .- JL_NESS)/JL_NESS) for JL in JL_set])
            title = "% diff between JL and JL_NESS"
        elseif choice == 2
            data =  list_to_matrix([100*((JR .- JR_NESS)/JR_NESS) for JR in JR_set])
            title = "% diff between JR and JR_NESS"
        elseif choice == 3
            data =  list_to_matrix([(100*(den .- den_NESS)/den_NESS) for den in den_set])
            title = "% diff between den and den_NESS"
        elseif choice == 4
            println("component analysis: k=1 for slowest mode, k>1 for others:")
            k = parse(Int,readline())

            data = abs.(getindex.(list_to_matrix(components_set),k+1))
            title = "Decay mode component"
        end
        return data,title,components_set

    end


    function map_to_positive(ρ,NESS)
        """
        For any ρ that's a decaying mode, the spectrum is symmetric about Re(λ)=0, so the
        both the negative and positive modes have equal overlap. 
        """
        ρ = unvectorise_ρ(ρ,false)
        vals,vecs = eigen(ρ)

        vals = real.(vals)
        neg_inds = findall(x->x<0,vals)
        pos_inds = findall(x->x>0,vals)

        ρ_pos = sum([vals[i]*kron(vecs[:,i],vecs[:,i]') for i in pos_inds])
        ρ_pos = ρ_pos/tr(ρ_pos)

        ρ_neg = sum([vals[i]*kron(vecs[:,i],vecs[:,i]') for i in neg_inds])
        ρ_neg = ρ_neg/tr(ρ_neg)

        ρ_neg_overlap = abs.(HS_norm(NESS,ρ_neg))
        ρ_pos_overlap = abs.(HS_norm(NESS,ρ_pos))
        
        ρ =ρ_neg
        if ρ_neg_overlap>ρ_pos_overlap
            ρ = ρ_pos
        end    
        #println("Overlap, negative vs positive: ",ρ_neg_overlap," vs ",ρ_pos_overlap)
        
        return vectorise_mat(ρ)
    end

    # function finding_p_max()
    #     """
    #     Inputs: p1_vec,p2_vec,p_th_vec,p_opt_vec
    #     Output: two vectors, one for the value of p_th at
    #     the maximum of Mpemba_vec(β), and another for the minimum.
    #     """
    #     Mpemba_vec = p_opt_vec-p_th_vec
    #     N1,N2 = size(Mpemba_vec)
        
    #     p_th_max = similar(p_th_vec[:,1])
    #     p_th_min = similar(p_th_vec[:,1])
        
    #     ###Should be β or Γ
    #     for i =1:N2
    #         _,ind1 = findmax(Mpemba_vec[:,i])
    #         _,ind2 = findmin(Mpemba_vec[:,i])
            
    #         p_th_max[i] = p_th_vec[ind1,i]
    #         p_th_min[i] = p_th_vec[ind2,i]
    #     end
    #     plot(p2_vec,p_th_max)
    #     display(plot!(p2_vec,p_th_min))
    # end

    function finding_μmax(Mpemba_tensor,μ_vec,pth_tensor,Γ_vec,β_vec)
        
        """
        Inputs: Mpemba tensor,pth_tensor,μ_vec.
        
        """
        Nμ,Nβ,NΓ = size(Mpemba_tensor)
        
        μmax_mat = similar(Mpemba_tensor[1,:,:])
        μmin_mat = similar(Mpemba_tensor[1,:,:])   
        pmax_mat = similar(Mpemba_tensor[1,:,:])
        pmin_mat = similar(Mpemba_tensor[1,:,:])
        labels = ["using Γ="*string(Γ) for Γ in Γ_vec]
        for i=1:Nβ
            for j = 1:NΓ
                val1,ind1 = findmax(Mpemba_tensor[:,i,j])
                val2,ind2 = findmin(Mpemba_tensor[:,i,j])
                
               # println("Γ="*string(Γ_vec[j])*",β="*string(β_vec[i]))
                #display(plot(μ_vec,Mpemba_tensor[:,i,j]))
                if val1 !=0
                    μmax_mat[i,j] = μ_vec[ind1]
                else
                    μmax_mat[i,j] = 0
                end
                μmin_mat[i,j] = μ_vec[ind2]
                pmax_mat[i,j] = pth_tensor[ind1,i,j]
                pmin_mat[i,j] = pth_tensor[ind2,i,j]
            end
        end
        plot_lines(μmax_mat,labels,β_vec,"β","μmax")
        plot_lines(pmax_mat,labels,β_vec,"β","pmax")
        
        plot_surface(μmax_mat,β_vec,Γ_vec,"β","Γ","μmax")
        plot_surface(μmin_mat,β_vec,Γ_vec,"β","Γ","μmin")
        plot_contour(pmax_mat,β_vec,Γ_vec,"β","μ","pmax")
        plot_contour(pmin_mat,β_vec,Γ_vec,"β","μ","pmax")
    end

    function L0_density(BP)
        γf = BP.Vk_fill_R[1]
        γe = BP.Vk_emp_R[1]
        den = (γf^2)/(γf^2 +γe^2)
        return den
    end

    function Effective_L_rates(L_vec,DP)

        Gd_eff = zeros(length(L_vec))
        Ge_eff = similar(Gd_eff)
        for i=1:length(L_vec)
            Gd_eff[i] =real(L_vec[i][1,2])
            Ge_eff[i] = real(L_vec[i][2,1])
        end
    
        data = [Gd_eff,Ge_eff]
        labels = ["Injection","Ejection"]
        plot_lines(data,labels,DP.times[2:end-1],"Time","Effective rates")
    end

    function propagate_p(p0,L_vec,P)

        decay_rates = [eigen(L).values[1] for L in L_vec]
        pths = [NESS_extraction(L,"L",P)[2][4] for L in L_vec]
        p_vec = complex(zeros(length(L_vec)+1))
        p_vec[1] = p0
        for i=1:length(L_vec)
            p_vec[i+1] = p_vec[i]+decay_rates[i]*(p_vec[i]-pths[i])*P.δt
        end
        return real.(p_vec)
    end

        
    function quench(corr,quench_time,β,β_prep,P,DP)
        """
        Input is the full correlation matrix of the prepared state.
        Setup: β=10,Γ=0.01,μ=0.2
        β_opt ~ 11.55
        """


        (;Nbr,Nbl,δt) = P
        (;q,N) = DP

        P.β_L,P.β_R = β_prep,β_prep
        DP = DP_initialisation(P)
        BP = BP_initialisation(DP,P)
        H_old = BP.H_single

        P.β_L,P.β_R = β,β
        DP = DP_initialisation(P)
        BP = BP_initialisation(DP,P)
        H_new = BP.H_single

        Nt = Int(round(quench_time/δt))
        t = range(P.δt,quench_time,Nt)
        Q_fun = tanh.(5*t/quench_time)

        left_inds = 1:2*Nbl
        right_inds = (N-2*Nbr+1):N
        corrs = Vector{Any}(undef,2*Nt+1)
        corrs[1] = corr
        for i=1:Nt
            fac = 1-Q_fun[i]
            H = copy(H_old)
            H[q,left_inds] *= fac 
            H[q,right_inds] *= fac
            H[left_inds,q] *= fac 
            H[right_inds,q] *= fac
            U_step = exp(-im*δt*H)
            corr = U_step*corr*U_step'
            corrs[i+1] = corr
        end

        ###The first bath has now been removed, so we also 
        ###remove the correlations with this bath.
        sys_corr = corr[q,q]
        corr = initial_correlation_matrix(0.5,DP,P)
        corr[q,q] = sys_corr

        for i=1:Nt
            fac = Q_fun[i]
            H = copy(H_new)
            H[q,left_inds] *= fac 
            H[q,right_inds] *= fac
            H[left_inds,q] *= fac 
            H[right_inds,q] *= fac
            U_step = exp(-im*δt*H)
            corr = U_step*corr*U_step'
            corrs[i+Nt+1] = corr
        end

        return corr,corrs,t
    end

    
function ρf_to_corr_with_ancilla(ρf,DP)
    
    (;N,q) = DP
    a = ρf[1,1]
    b = ρf[2,2]
    c = ρf[2,3]
    d = ρf[4,4]

    ###build ψ
    α = √(a)
    β = √((b-√(b^2-c^2))/2)
    γ = √((b+√(b^2-c^2))/2)
    ω = √(d)
    
    qN = extract_physical_modes(2)
    ψ = complex(zeros(16))
    ψ[qN[1]] = α
    ψ[qN[2]] = γ
    ψ[qN[3]] = β
    ψ[qN[4]] = β
    ψ[qN[5]] = conj(γ)
    ψ[qN[6]] = ω
    x = ψ*ψ'
    corr = ρ_system_corr(x,Ns)

    C = complex(zeros(N,N))
    [C[i,i] = isodd(i) ? 1 : 0 for i=1:N]
    C[q,q] = corr

    return C
end
function dpdt(p_vec,t)
    Np = length(p_vec)
    dpdt = zeros(Np-2)
    dt = t[2]-t[1]
    for i = 1:length(dpdt) 
        dpdt[i] = (p_vec[i+1]-p_vec[i])/dt
    end
    Plots.display(Plots.plot(t[3:end-2],dpdt))
    return dpdt
end

