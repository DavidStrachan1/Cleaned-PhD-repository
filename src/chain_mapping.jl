module chain_mapping


    using ITensors
    using ITensorMPS
    using PolyChaos
    using LinearAlgebra


    export BathParameters
    export SystemParameters
    export RingParameters
    export TDVP_parameters
    export ThermofieldSector
    export Filled
    export Empty
    export NEQ_ChainLayout
    export EQ_ChainLayour
    export fermi_factor
    export heaviside
    export semicircular_density
    export box_spectral_density
    export spectral_density
    export thermofield_spectral_density
    export create_chain_coeffs
    export chainLayout
    export add_chain
    export couple_sites
    export build_NEQ_fermi_chain_model
    export build_thermofield_ring_model



    Base.@kwdef struct BathParameters
        Γ::Float64 #Coupling to system
        β::Float64 #inverse temperature
        μ::Float64 #chemical potential
        D::Float64 #bandwidth
        N::Int     #Number of chain modes
        spectral_function::String
    end

    Base.@kwdef struct SystemParameters
        ϵ::Vector{Float64} #Vector of onsite energies
        t::Vector{Float64} #Vector of couplings
        U::Vector{Float64} #Vector of interactions
        occupations::Vector{String} # Vector of initial occupations
        compute_maps_bool
    end

    Base.@kwdef struct RingParameters
        J::Float64 #Vector of onsite energies
        B::Float64 #Vector of couplings
        N_ring::Int #Vector of interactions
        occupations::Vector{Float64} # Vector of initial occupations
        compute_maps_bool::Bool
    end

    Base.@kwdef struct TDVP_parameters
        tdvp_cutoff::Float64 #Numerical cutoff for tdvp
        minbonddim::Int      #Minimum bond dimension for tdvp
        maxbonddim::Int      #Maximum bond dimension for tdvp
        δt::Float64          #Time step
        total_simulation_time::Float64 #Evolution time
    end

    abstract type ThermofieldSector end
    struct Filled <: ThermofieldSector end
    struct Empty <: ThermofieldSector end

    struct NEQ_ChainLayout

        left_filled
        left_empty

        system

        right_filled
        right_empty
    end

    struct EQ_ChainLayout
        system
        filled_bath
        empty_bath
    end

    fermi_factor(ω,β,μ) = 1 / (exp(β*(ω-μ)) + 1)
    heaviside(t) = 0.5 * (sign.(t) .+ 1)


    function semicircular_density(Γ,ω,D)
        J = real((2*Γ/(π^2))*sqrt.(Complex.(1 .-(ω/D).^2)))
        return J

    end

    function box_spectral_density(Γ,ω,D)
        #Box spectral density
        return J = (Γ/(2*π))*(heaviside(ω .+ D) .- heaviside(ω .- D))
    end

    function smoothed_box_spectral_density(Γ,ω,D)
        ν =  10
        denominator = (1 .+exp.(ν*(ω .-D))).*(1 .+exp.(-ν*(ω .+D)))
        J = (Γ/(2*π)) ./denominator
        return J
    end
    function spectral_density(Γ,ω,D,choice)
        if choice == "box"
            J= box_spectral_density(Γ,ω,D)
        elseif choice == "ellipse"
            J = semicircular_density(Γ,ω,D)
        elseif choice == "smoothed box"
            J = smoothed_box_spectral_density(Γ,ω,D)
        end
        return J
    end

    function thermofield_spectral_density(ω,bath::BathParameters,::Filled)
        #Spectral density for a filled chain
        J = spectral_density(bath.Γ,ω,bath.D,bath.spectral_function)
        return J * fermi_factor(ω,bath.β,bath.μ)
    end

    function thermofield_spectral_density(ω,bath::BathParameters,::Empty)
        #Spectral density for an empty chain
        J = spectral_density(bath.Γ,ω,bath.D,bath.spectral_function)
        return J * (1 - fermi_factor(ω,bath.β,bath.μ))
    end

    function create_chain_coeffs(bath::BathParameters,sector::ThermofieldSector)

        #Calculates the chain coefficients using PolyChaos.jl

        spec_fun(ω) = thermofield_spectral_density(ω,bath,sector)

        #support needs to be larger than spectral function for numerical reasons.
        support = (-2*bath.D,2*bath.D)

        meas = Measure("thermofield",spec_fun,support,false,Dict())
        op = OrthoPoly("chain",bath.N-1,meas;Nquad=100000)

        α = coeffs(op)[:,1]
        β = coeffs(op)[:,2]

        return α,sqrt.(β)
    end

    """
    Two versions of chainLayout, add_chain, couple sites and build hamiltonian
    """

    function ChainLayout(N_bath,Nsys)
        ##Arranges the chains in interleaved fashion, with the system at the start. Any
        ##other layout can be encoded here and will follow through to the rest of the code.


        N = 2*N_bath+Nsys

        EQ_ChainLayout(
            1:Nsys,
            Nsys+1:2:N,
            Nsys+2:2:N,
        )
    end

    function ChainLayout(N_left_bath,N_right_bath,Nsys)
        ##Arranges the chains in interleaved fashion, with the system at the centre. Any
        ##other layout can be encoded here and will follow through to the rest of the code.

        N = 2*(N_left_bath+N_right_bath) + Nsys

        NEQ_ChainLayout(
            1:2:2*N_left_bath,
            2:2:2*N_left_bath,
            2*N_left_bath+1:2*N_left_bath+Nsys,
            2*N_left_bath+Nsys+1:2:N,
            2*N_left_bath+Nsys+2:2:N,
        )
    end

    function add_chain(H,os,inds,energies,hoppings)
        ##Adds MPO terms and associated single particle hamiltonian elements 
        ##for the thermofield chain

        N = length(inds)
        for i in 1:N
            os += energies[i],"N",inds[i]

            H[inds[i],inds[i]] = energies[i]

            if i < N
                t = hoppings[i]
                os += t,"Cdag",inds[i],"C",inds[i+1]
                os += t,"Cdag",inds[i+1],"C",inds[i]

                H[inds[i],inds[i+1]] = t
                H[inds[i+1],inds[i]] = t
            end
        end
        return H,os
    end

    function add_chain(H,inds,energies,hoppings)
        ##Adds MPO terms and associated single particle hamiltonian elements 
        ##for the thermofield chain

        N = length(inds)
        for i in 1:N
            #os += energies[i],"N",inds[i]

            H[inds[i],inds[i]] = energies[i]

            if i < N
                t = hoppings[i]
            #   os += t,"Cdag",inds[i],"C",inds[i+1]
            #  os += t,"Cdag",inds[i+1],"C",inds[i]

                H[inds[i],inds[i+1]] = t
                H[inds[i+1],inds[i]] = t
            end
        end
        return H#,os
    end

    function couple_sites(H,i,j,t)
        #Couples two sites, used for the system-chain coupling

        # os += t,"Cdag",i,"C",j
        #os += t,"Cdag",j,"C",i

        H[i,j] = t
        H[j,i] = t
        return H#,os
    end

    function couple_sites(H,os,i,j,t)
        #Couples two sites, used for the system-chain coupling

        os += t,"Cdag",i,"C",j
        os += t,"Cdag",j,"C",i

        H[i,j] = t
        H[j,i] = t
        return H,os
    end

    function build_NEQ_fermi_chain_model(sites,left,right,sys)
        #builds single particle hamiltonian and many body MPO

        layout =ChainLayout(left.N,right.N,length(sys.ϵ))
        N = length(sites)
        Hsingle = zeros(ComplexF64,N,N)
        os = OpSum()

        #chain coefficients for the four chains
        εLF,tLF =create_chain_coeffs(left,Filled())
        εLE,tLE =create_chain_coeffs(left,Empty())
        εRF,tRF =create_chain_coeffs(right,Filled())
        εRE,tRE =create_chain_coeffs(right,Empty())

        #adds the terms for the chains
        Hsingle,os = add_chain(Hsingle,os,layout.left_filled,reverse(εLF),reverse(tLF))
        Hsingle,os = add_chain(Hsingle,os,layout.left_empty,reverse(εLE),reverse(tLE))
        Hsingle,os = add_chain(Hsingle,os,layout.right_filled,εRF,tRF[2:end])
        Hsingle,os = add_chain(Hsingle,os,layout.right_empty,εRE,tRE[2:end])


        #
        # system
        #

        for i in eachindex(sys.ϵ)
            ##energies
            os += sys.ϵ[i],"N",layout.system[i]
            Hsingle[layout.system[i],layout.system[i]] = sys.ϵ[i]
            if i < length(sys.ϵ)
                #hoppings
                Hsingle,os = couple_sites(Hsingle,os,layout.system[i],layout.system[i+1],sys.t[i])
                #interactions
                os += sys.U[i],"N",layout.system[i],"N",layout.system[i+1]
            end
        end

        #
        # bath-system couplings
        #

        Hsingle,os = couple_sites(Hsingle,os,last(layout.left_filled),first(layout.system),first(tLF))
        Hsingle,os = couple_sites(Hsingle,os,last(layout.left_empty),first(layout.system),first(tLE))
        Hsingle,os = couple_sites(Hsingle,os,first(layout.right_filled),last(layout.system),first(tRF))
        Hsingle,os = couple_sites(Hsingle,os,first(layout.right_empty),last(layout.system),first(tRE))
        return MPO(os,sites), Hsingle
    end

    function build_thermofield_ring_model(bath,sys)
        #builds single particle hamiltonian and single particle correlation matrix

        if sys.compute_maps_bool
            Nsys = 2*sys.N_ring
            qS = 1:sys.N_ring
            qA = sys.N_ring+1:2*sys.N_ring
            q = 1:2*sys.N_ring
        else
            Nsys = sys.N_ring
            qS = 1:sys.N_ring
            qA = 0:0
            q = qS
        end

        layout =ChainLayout(bath.N,Nsys)

        N = 2*bath.N+Nsys
        Hsingle = zeros(ComplexF64,N,N)
        initial_correlation_matrix = zeros(ComplexF64,N,N)

        #chain coefficients for the two chains
        εF,tF =create_chain_coeffs(bath,Filled())
        εE,tE =create_chain_coeffs(bath,Empty())

        #adds the terms for the chains
        Hsingle = add_chain(Hsingle,layout.filled_bath,εF,tF[2:end])
        Hsingle = add_chain(Hsingle,layout.empty_bath,εE,tE[2:end])

        #system terms
        for i =1:sys.N_ring
            Hsingle[qS[i],qS[i]] = -2*sys.B
            if i <sys.N_ring
                Hsingle[qS[i+1],qS[i]] = -2*sys.J
                Hsingle[qS[i],qS[i+1]] = -2*sys.J
            elseif i!= 1
                @assert(i==sys.N_ring)
                Hsingle[qS[1],qS[i]] = -2*sys.J
                Hsingle[qS[i],qS[1]] = -2*sys.J
            end
        end


        #
        # bath-system couplings
        #

        Hsingle = couple_sites(Hsingle,first(layout.filled_bath),last(qS),first(tF))
        Hsingle = couple_sites(Hsingle,first(layout.empty_bath),last(qS),first(tE))


        N = 2*bath.N+Nsys
        initial_C = zeros(ComplexF64,N,N)

        #create initial correlation matrix
        [initial_C[i,i] = 1 for i in layout.filled_bath]
        if sys.compute_maps_bool
            for i =1:sys.N_ring 
                initial_C[qS[i],qS[i]] = 0.5
                initial_C[qS[i],qA[i]] = 0.5
                initial_C[qA[i],qS[i]] = 0.5
                initial_C[qA[i],qA[i]] = 0.5
            end
        else
            for i =1:sys.N_ring 
                initial_C[qS[i],qS[i]] = sys.occupations[i]
            end
        end
        return Hsingle,initial_C,q,qS,qA
    end


end