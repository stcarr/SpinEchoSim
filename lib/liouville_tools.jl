# see :https://arxiv.org/abs/1510.08634
#       -> Sec. 1 (Vec-ing) is in column major format, so easy implementation in Julia

module LiouvilleTools

    using StaticArrays
    using LinearAlgebra


    # convert Hamiltonian from Hilbert space to Liouville space
    function HamToSuper(H)
        # takes: H, the n x n Hamiltonian acting on the Hilbert space of a spinor
        # returns: the Liouvillian, L, which can be used to generate the
        # time propagator for the spinor's density matrix as follows:
        # |ρ(t)> = exp[-1i*L*dt]|ρ(0)>

        n = size(H,1)
        Id = SMatrix{n,n}(I)
        L = kron(Id,H) - kron(conj(H),Id) # used H hermitian to convert H^T -> H*

        return L

    end

    # makes the superoperator for dissipation, given set of Linblad jump operators
    function JumpsToSuper(Lj_list)
        # takes:   Lj_list, an nJ x (n x n) list of jump operators
        # returns: J_tot,   an n^2 x n^2 superoperator of all jumps

        n = size(Lj_list,2) # dimension of system
        if (n > 0)
            J_tot = JumpToSuper(Lj_list[1])

            nJ = size(Lj_list,1) # number of jump ops.
            for j = 2:nJ
                J_tot = J_tot + JumpToSuper(Lj_list[j])
            end
            return J_tot
        else
            return nothing
        end
    end

    # convert Hilbert-space non-Hermitian jump operator to a dissipative Liouville superoperator
    function JumpToSuper(J)
        # takes:   J,   an  n   x n   jump operator in the Hilbert space of a spinor
        # returns: L_J, an  n^2 x n^2 superoperator acting on the Liouville space of ρ
        n = size(J,1)
        Id = SMatrix{n,n}(I)
        L_J = kron(transpose(J'), J) - 0.5*( kron(Id,J'*J) + kron( transpose(J'*J) ,Id) )
        return L_J 
    end

    # convert density matrix from Hilbert to Liouville space
    function dm_H2L(ρ_H)
        # takes:   ρ_H, an n   x n density matrix in Hilbert space
        # returns: ρ_L, an n^2 x 1 density matrix in Liouville space
        n = size(ρ_H,1)
        return reshape( ρ_H, n^2)
    end

    # convert density matrix from Liouville to Hilbert space
    function dm_L2H(ρ_L)
        # takes:   ρ_L, an n^2 x 1 density matrix in Liouville space
        # returns: ρ_H, an n   x n density matrix in Hilbert space
        n = convert(Int64, sqrt(size(ρ_L,1)))
        return reshape(ρ_L,n,n)
    end

    # convert left-acting operator (e.g. Aρ) from Hilbert to Liouville space
    function leftop_H2L(A)
        # takes:   A,   an n   x n operator in Hilbert space
        # returns: A_L, an n^2 x 1 operator in Liouville space
        n = size(A,1)
        Id = SMatrix{n,n}(I)
        return kron(Id,A)

    end

    # convert right-acting operator (e.g. ρB)  from Hilbert to Liouville space
    function rightop_H2L(B)
        # takes:   B,   an n   x n operator in Hilbert space
        # returns: B_L, an n^2 x 1 operator in Liouville space
        n = size(B,1)
        Id = SMatrix{n,n}(I)
        return kron(transpose(B),Id)

    end

    # get trace of density matrix  (when in Liouville basis)
    function tr_L(ρ_L)
        n = convert(Int64, sqrt(size(ρ_L,1)))
        sum = 0
        for i = 1:n
            sum = sum + ρ_L[i + n*(i-1)] # gets what would be diagonal elements in n x n basis
        end
        return sum

    end

    export HamToSuper, JumpsToSuper, JumpToSuper
    export dm_H2L, dm_L2H, leftop_H2L, rightop_H2L, tr_L

end