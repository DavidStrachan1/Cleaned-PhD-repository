module Setup

    using ITensors
    using ITensorMPS
    using PolyChaos
    using LinearAlgebra


    export BathParameters
    export SystemParameters
    export TDVP_parameters
    export ThermofieldSector
    export fermi_factor
    export heaviside
    export semicircular_density
    export box_spectral_density
    export thermofield_spectral_density
    export chain_mapping
    export ChainLayout
    export add_chain
    export couple_sites
    export build_hamiltonian
    export thermofield_state
    export current_time
    export measure_correlation_matrix
    export particle_current
    export density

    Base.@kwdef struct BathParameters
        Γ::Float64 #Coupling to system
        β::Float64 #inverse temperature
        μ::Float64 #chemical potential
        D::Float64 #bandwidth
        N::Int     #Number of chain modes
    end

    Base.@kwdef struct SystemParameters
        ϵ::Vector{Float64} #Vector of onsite energies
        t::Vector{Float64} #Vector of couplings
        U::Vector{Float64} #Vector of interactions
        occupations::Vector{String} # Vector of initial occupations
    end

    Base.@kwdef mutable struct TDVP_parameters
        tdvp_cutoff::Float64 #Numerical cutoff for tdvp
        minbonddim::Int      #Minimum bond dimension for tdvp
        maxbonddim::Int      #Maximum bond dimension for tdvp
        δt::Float64          #Time step
        total_simulation_time::Float64 #Evolution time
    end


    abstract type ThermofieldSector end
    struct Filled <: ThermofieldSector end
    struct Empty <: ThermofieldSector end

    struct ChainLayout

        left_filled
        left_empty

        system
        ancilla

        right_filled
        right_empty
    end


    """
    Initialisation functions
    """

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

    function thermofield_spectral_density(ω,bath::BathParameters,::Filled)
        #Spectral density for a filled chain
       # J = box_spectral_density(bath.Γ,ω,bath.D)
        J = semicircular_density(bath.Γ,ω,bath.D)
        return J * fermi_factor(ω,bath.β,bath.μ)
    end

    function thermofield_spectral_density(ω,bath::BathParameters,::Empty)
        #Spectral density for an empty chain
       # J = box_spectral_density(bath.Γ,ω,bath.D)
        J = semicircular_density(bath.Γ,ω,bath.D)
        return J * (1 - fermi_factor(ω,bath.β,bath.μ))
    end

    function chain_mapping(bath::BathParameters,sector::ThermofieldSector)

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

    function ChainLayout(N_left_bath,N_right_bath,Nsys)
        ##Arranges the chains in interleaved fashion, with the system at the centre. Arranges 
        #the system and system ancillas in separated fashion.Any
        ##other layout can be encoded here and will follow through to the rest of the code.

        N = 2*(N_left_bath+N_right_bath) + 2*Nsys

        ChainLayout(
            1:2:2*N_left_bath,
            2:2:2*N_left_bath,
            2*N_left_bath+1:2*N_left_bath+Nsys,
            2*N_left_bath+Nsys+1:2*N_left_bath+2*Nsys,
            2*N_left_bath+2*Nsys+1:2:N,
            2*N_left_bath+2*Nsys+2:2:N,
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

    function couple_sites(H,os,i,j,t)
        #Couples two sites, used for the system-chain coupling

        os += t,"Cdag",i,"C",j
        os += t,"Cdag",j,"C",i

        H[i,j] = t
        H[j,i] = t
        return H,os
    end

    function build_hamiltonian(sites,left,right,sys)
        #builds single particle hamiltonian and many body MPO

        layout =ChainLayout(left.N,right.N,length(sys.ϵ))
        N = length(sites)
        Hsingle = zeros(ComplexF64,N,N)
        os = OpSum()

        #chain coefficients for the four chains
        if left.N >0
            εLF,tLF =chain_mapping(left,Filled())
            εLE,tLE =chain_mapping(left,Empty())
        end
        εRF,tRF =chain_mapping(right,Filled())
        εRE,tRE =chain_mapping(right,Empty())

        #adds the terms for the chains
        if left.N>0
            Hsingle,os = add_chain(Hsingle,os,layout.left_filled,reverse(εLF),reverse(tLF))
            Hsingle,os = add_chain(Hsingle,os,layout.left_empty,reverse(εLE),reverse(tLE))
        end
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

        if left.N>0
            Hsingle,os = couple_sites(Hsingle,os,last(layout.left_filled),first(layout.system),first(tLF))
            Hsingle,os = couple_sites(Hsingle,os,last(layout.left_empty),first(layout.system),first(tLE))
        end
        Hsingle,os = couple_sites(Hsingle,os,first(layout.right_filled),last(layout.system),first(tRF))
        Hsingle,os = couple_sites(Hsingle,os,first(layout.right_empty),last(layout.system),first(tRE))
        return MPO(os,sites), Hsingle
    end

    function thermofield_state(left,right,system)
        #defines the initial state given both baths have an interleaved ordering
        #for the filled and empty chains. Both start with the filled mode from the left.
        #The system and ancilla are initialised in the empty states.

        layout =ChainLayout(left.N,right.N,length(system.ϵ))
        N = 2*(left.N+right.N)+2*length(system.ϵ)
        
        occs = Vector{String}(undef,N)
        if left.N>0
            occs[layout.left_filled] .= "Occ"
            occs[layout.left_empty] .= "Emp"
        end
        occs[layout.right_filled] .= "Occ"
        occs[layout.right_empty] .="Emp"
        occs[layout.system] .= "Emp"
        occs[layout.ancilla] .= "Emp" 

        return occs
    end

    #Evolution functions

    function current_time(; current_time, bond, half_sweep)

        if bond == 1 && half_sweep == 2
        return real(im*current_time)
        end
        return nothing
    end
    function measure_correlation_matrix(; state, bond, half_sweep)
        if bond==1 && half_sweep == 2
            return correlation_matrix(state,"Cdag","C")
        end
        return nothing
    end

    function particle_current(site,corr,hopping_term)
        #measures the current from site to site+1
        return 2*hopping_term*imag(corr[site+1,site])
    end
    function density(site,corr)
        #measures density at site
        return real(corr[site,site])
    end

end