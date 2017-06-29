# A simple Jacobi iteration

Project of Parallel and Concurrent Programming on the Cloud course.
Professor: Vittorio Scarano

### Problem statement
In this example, you will put together some of the previous examples to implement a simple Jacobi iteration for approximating the solution to a linear system of equations. In this example, we solve the Laplace equation in two dimensions with finite differences. This may sound involved, but really amount only to a simple computation, combined with the previous example of a parallel mesh data structure.

Any numerical analysis text will show that iterating
```
while (not converged) {
  for (i,j)
    xnew[i][j] = (x[i+1][j] + x[i-1][j] + x[i][j+1] + x[i][j-1])/4;
  for (i,j)
    x[i][j] = xnew[i][j];
  }
  ```
  
will compute an approximation for the solution of Laplace's equation. There is one last detail; this replacement of xnew with the average of the values around it is applied only in the interior; the boundary values are left fixed. In practice, this means that if the mesh is n by n, then the values
```
x[0][j]
x[n-1][j]
x[i][0]
x[i][n-1]
```
are left unchanged. Of course, these refer to the complete mesh; you'll have to figure out what to do with for the decomposed data structures (xlocal).
Because the values are replaced by averaging around them, these techniques are called relaxation methods.

We wish to compute this approximation in parallel. Write a program to apply this approximation. For convergence testing, compute
```
diffnorm = 0;
for (i,j)
    diffnorm += (xnew[i][j] - x[i][j]) * (xnew[i][j] - x[i][j]);
diffnorm = sqrt(diffnorm);
```

When diffnorm is less that 1.0e-2, consider the iteration converged. Also, if you reach 100 iterations, exit the loop.

For simplicity, consider a 12 x 12 mesh on 4 processors. 
The example solution uses the boundary values from the previous exercise; they are -1 on the top and bottom, and the rank of the process on the side. The initial data (the values of x that are being relaxed) are also the same; the interior points have the same value as the rank of the process. This is shown below:

```
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 
 3  3  3  3  3  3  3  3  3  3  3  3
 3  3  3  3  3  3  3  3  3  3  3  3
 2  2  2  2  2  2  2  2  2  2  2  2
 2  2  2  2  2  2  2  2  2  2  2  2
 2  2  2  2  2  2  2  2  2  2  2  2
 1  1  1  1  1  1  1  1  1  1  1  1
 1  1  1  1  1  1  1  1  1  1  1  1
 1  1  1  1  1  1  1  1  1  1  1  1
 0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 
```

Note that this is a very poor way to solve this numerical problem, and this method is being used only because it is very simple. Fortunately, the MPI parts of this example are very similar to those that are used in the better parallel algorithms for this problem. In particular, the use of ghost points in the parallel data structure is very similar to what is used in methods such as Conjugate Gradient or Multigrid.



### Benchmarking

1) Provide a solution to the problem exploiting parallel computation and develop  a C program using MPI. The provided implementation can use either Point-to-Point communication or Collective communication routines.
2) Benchmark the solution on Amazon AWS (EC2) on General Purpose instances (e.g. M3.medium family) or on Compute optimize instances (e.g. C3.large family).  Testing the solution using 1, 2, 3, 4, 5, 6, 7, 8 instances.
3) Both weak and strong scalability have to be analyzed:
- Strong Scaling: Keeping the problem size fixed and pushing in more workers or processors. Goal: Minimize time to solution for a given problem.
- Weak Scaling: Keeping the work per worker fixed and adding more workers/processors (the overall problem size increases). Goal: solve the larger problems.

###### HINT

1) The results should be presented as two different scatter x-y charts, where the x-axis denotes the number of MPI processors used and the y-axis value represents the time in milliseconds.  
2) The number of MPI processors should be equal to the number of cores.
