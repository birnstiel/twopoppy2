! _____________________________________________________________________________
! Implicit donor cell advection-diffusion scheme with piecewise constant values
!
!     Perform one time step for the following PDE:
!
!        du    d /     \    d /              d  /     u  \ \
!        -- + -- | u v | - -- | h(x) Diff(x) -- |g(x)----| | = K + L u
!        dt   dx \     /   dx \              dx \    h(x)/ /
!
!     with boundary conditions
!
!         dgu/h |            |
!       p ----- |      + q u |       = r
!          dx   |x=xbc       |x=xbc
! INPUT:
!       n_x     = # of grid points
!       x()     = the grid
!       Diff()  = value of Diff @ cell center
!       v()     = the values for v @ interface (array(i) = value @ i-1/2)
!       g()     = the values for g(x)
!       h()     = the values for h(x)
!       K()     = the values for K(x)
!       L()     = the values for L(x)
!       u()     = the current values of u(x)
!       dt      = the time step
!
! OUTPUT:
!       u()     = the updated values of u(x) after timestep dt
!
! NEEDS:
!       subroutine  tridag(a,b,c,r,u,n)         to invert tridiagonal matrix
! _____________________________________________________________________________
subroutine impl_donorcell_adv_diff_delta(n_x,x,Diff,v,g,h,K,L,u,dt,pl,pr,ql,qr,rl,rr, u_out)
    implicit none

    integer,intent(in)             :: n_x
    doubleprecision, intent(in)    :: x(1:n_x),Diff(1:n_x),g(1:n_x),h(1:n_x),K(1:n_x),L(1:n_x)
    doubleprecision, intent(in)    :: v(1:n_x) ! array(n) = value @ n-1/2
    doubleprecision, intent(inout) :: u(1:n_x)
    doubleprecision, intent(in)    :: dt
    doubleprecision, intent(out)   :: u_out(1:n_x)
    doubleprecision :: A(1:n_x),B(1:n_x),C(1:n_x),D(1:n_x)
    doubleprecision :: rhs(1:n_x),u2(1:n_x)
    doubleprecision :: D05(1:n_x),h05(1:n_x),vol
    doubleprecision :: pl,pr,ql,qr,rl,rr
    integer :: i

    ! ----- calculate the arrays at the interfaces
    do i = 2,n_x
        D05(i) = 0.5d0 * (Diff(i-1) + Diff(i))
        h05(i) = 0.5d0 * (h(i-1) + h(i))
    enddo

    ! ----- calculate the entries of the tridiagonal matrix
    do i = 2,n_x-1
        vol = 0.5d0*(x(i+1)-x(i-1))
        A(i) = -dt/vol *  &
            & ( &
            & + max(0.d0,v(i))  &
            & + D05(i) * h05(i) * g(i-1) / (  (x(i)-x(i-1)) * h(i-1)  ) &
            & )
        B(i) = 1.d0 - dt*L(i) + dt/vol * &
            & ( &
            & + max(0.d0,v(i+1))   &
            & - min(0.d0,v(i))  &
            & + D05(i+1) * h05(i+1) * g(i)   / (  (x(i+1)-x(i)) * h(i)    ) &
            & + D05(i)   * h05(i)   * g(i)   / (  (x(i)-x(i-1)) * h(i)    ) &
            & )
        C(i) = dt/vol *  &
            & ( &
            & + min(0.d0,v(i+1))  &
            & - D05(i+1) * h05(i+1)  * g(i+1) / (  (x(i+1)-x(i)) * h(i+1)  ) &
            & )
        D(i) = -dt * K(i)
    enddo

    ! ----- boundary Conditions
    A(1)   = 0.d0
    B(1)   = ql - pl*g(1) / (h(1)*(x(2)-x(1)))
    C(1)   =      pl*g(2) / (h(2)*(x(2)-x(1)))
    D(1)   = u(1)-rl

    A(n_x) =    - pr*g(n_x-1) / (h(n_x-1)*(x(n_x)-x(n_x-1)))
    B(n_x) = qr + pr*g(n_x)  / (h(n_x)*(x(n_x)-x(n_x-1)))
    C(n_x) = 0.d0
    D(n_x) = u(n_x)-rr
        
    ! the delta-way
    
    do i = 2,n_x-1
        rhs(i) = u(i) - D(i) - (A(i)*u(i-1)+B(i)*u(i)+C(i)*u(i+1))
    enddo
    rhs(1)   = rl - (                B(1)*u(1)     + C(1)*u(2))
    rhs(n_x) = rr - (A(n_x)*u(n_x-1)+B(n_x)*u(n_x)            )

    ! solve for u2
    
    call tridag(A,B,C,rhs,u2,n_x)

    ! update u
    u_out = u + u2  ! delta way

end subroutine impl_donorcell_adv_diff_delta


! _____________________________________________________________________________
! the tridag routine from Numerical Recipes in F77 rewritten to F95
!
! where:    a         =    lower diagonal entries
!            b        =    diagonal entries
!            c        =    upper diagonal entries
!            r        =    right hand side vector
!            u        =    result vector
!            n        =    size of the vectors
! _____________________________________________________________________________
subroutine tridag(a,b,c,r,u,n)
    integer        :: n
    doubleprecision    :: a(n),b(n),c(n),r(n),u(n)
    integer, parameter :: NMAX=10000000
    doubleprecision    :: bet,gam(NMAX)
    integer :: j


    if (b(1).eq.0.) stop 'tridag: rewrite equations'

    bet = b(1)

    u(1)=r(1)/bet

    do j=2,n
        gam(j)    = c(j-1)/bet
        bet    = b(j)-a(j)*gam(j)
        if(bet.eq.0.) stop 'tridag failed'
        u(j)    = (r(j)-a(j)*u(j-1))/bet
    enddo

    do j=n-1,1,-1
        u(j)=u(j)-gam(j+1)*u(j+1)
    enddo
end subroutine tridag
! =============================================================================
