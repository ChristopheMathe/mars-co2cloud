program precision
  implicit none

  real :: a
  real(kind=8) :: b
  real(kind=16) :: c
  double precision :: e
   ! -- tiny and huge grab the smallest and largest
   ! -- representable number of each type
   write(*,*) 'Range for REAL:  ',  tiny(a),  huge(a)
   write(*,*) 'Range for REAL8:  ',  tiny(b),  huge(b)
   write(*,*) 'Range for REAL16:  ',  tiny(c),  huge(c)
   write(*,*) 'Range for DOUBLE PRECISION: ', tiny(e), huge(e)

   a = 1e-16
   b = 1e-16
   c = 1e-16
   e = 1e-16
   write(*,*) 'for REAL:             ',  a
   write(*,*) 'for REAL8:            ', b
   write(*,*) 'for REAL16:           ',  c
   write(*,*) 'for DOUBLE PRECISION: ', e

end program precision