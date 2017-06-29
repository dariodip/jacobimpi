<h1 id="jacobi">JACOBI</h1>
<h2 id="introduzione">Introduzione</h2>
<p>In analisi numerica, il <strong>metodi di Jacobi</strong> è un metodo iterativo per la risoluzione di sistemi lineari: calcola la soluzione di un sistema di equazioni lineari dopo un numero teoricamente infinito di passi.<br>
Il metodo utilizza una successione \(\textbf{x}^{(k)}\) che converge verso la soluzione esatta del sistema lineare e ne calcola progressivamente i valori arrestandosi quando la soluzione ottenuta è sufficientemente vicina a quella esatta.</p>
<h3 id="algoritmo">Algoritmo</h3>
<p>Sia \(Ax=b\) il sistema lineare da risolvere. Scriviamo a come \(A = M - N\), dove \(M\) è una matrice invertibile. A questo punto è possibile ottenere una soluzione per \(x\) risolvendo:</p>
<ul>
<li>\(Mx = Nx + b\);</li>
<li>\(x = M^{-1}(Nx+b)\).<br>
Partendo da un qualunque vettore \(x_0\), si può costruire una successione di vettori \(x_k\) come:<br>
\(x^{k+1} = M ^{-1}(Nx ^{k}+b)\).<br>
Se questa successione converge ad \(x\), allora \(Ax = b\).</li>
</ul>
