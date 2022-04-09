using PyPlot, LinearAlgebra, Roots, DifferentialEquations

function f(u,v; b=0.10, i=0.00)
	return u*(1.0-u)*(u-b) - v + i
end

function g(u,v; a=0.37, c=0.05)
	return c*(a*u-v)
end

function FHNgrid(; a=0.37, b=0.1, c=0.05, i=0.0, NU=1025, NV=1025, ulims=[-2.,+2.],vlims=[-1.,+1.])
	uu = range(ulims[1],ulims[2],NU)
	vv = range(vlims[1],vlims[2],NV)
	U = zeros(Float64, NV, NU)
	V = zeros(Float64, NV, NU)
	F = zeros(Float64, NV, NU)
	G = zeros(Float64, NV, NU)
	
	for m in 1:NV, n in 1:NU
		U[m,n] = uu[n]
		V[m,n] = vv[m]
		F[m,n] = f(U[m,n],V[m,n];b=b,i=i)
		G[m,n] = g(U[m,n],V[m,n];a=a,c=c)
	end
	
	return U,V,F,G
end

function FHNnullclines(; a=0.37, b=0.1, c=0.05, i=0.0)

	U,V,F,G = FHNgrid(; a=a, b=b, c=c, i=i)
	fig = plt.contour(U,V,F,levels=[-1e-14,+1e-14],colors="C0",linestyles="solid")
	plt.contour(U,V,G,levels=[-1e-14,+1e-14],colors="C1",linestyles="solid")
	plt.xlabel("\$ u \$")
	plt.ylabel("\$ v \$")
	plt.xlim([-2,2])
	plt.ylim([-1,1])
	return fig
end

function FHNequilibria(; a=0.37, b=0.1, c=0.05, i=0.0)

	# need to find ALL u s.t.:
	#	f(u,v) = 0 && g(u,v) = 0
	#	Since g(u,v) = 0 is linear, we invert to get v = a*u
	# plugging this into f(u,v=a*u) = 0, we have a single
	# scalar, cubic, equation to find the roots of.
	# There will be between 0 and 3 roots.
	ulim = 2.0
	r = find_zeros((u)->f(u,a*u;b=b,i=i),-ulim,ulim)
	while length(r) < 1 && ulim < 100.0
		ulim = 2*ulim
		r = find_zeros((u)->f(u,a*u;b=b,i=i),-ulim,ulim)
	end
	return reduce(hcat, (r,a*r))
	
	# NOTE: for the general case, where g(u,v) = 0 is not invertible,
	# you must use Newton's method:
	#	[u;v] = [u;v] - FHNjac(u,v)\[f(u,v);g(u,v)]
	# in a loop, until convergence (u & v do not change on successive iterations)
	# AND this only works for one root at a time, whereas we need all three!
	# AND we need good estimates to initialize this Newton iteration!
	#u = randn(); v = randn()
	#for n in 1:20
	#	(u,v) = (u,v) .- FHNjac(u,v; a=a,b=b,c=c,i=i)\[f(u,v;b=b,i=i);g(u,v;a=a,c=c)]
	#end
end

function FHNjac(u,v; a=0.37, b=0.1, c=0.05, i=0.0, h=1e-14)
	# this uses Complex-Step differentiation:
	# 	https://blogs.mathworks.com/cleve/2013/10/14/complex-step-differentiation/
	# a method of computing derivatives efficiently and accurately
	# using complex arithmetic.
	# It may, however, fail if the functions are not holonomic!
	J = zeros(Float64,2,2)
	J[1,1] = imag(f(u+im*h,v;b=b,i=i))/h
	J[1,2] = imag(f(u,v+im*h;b=b,i=i))/h
	J[2,1] = imag(g(u+im*h,v;a=a,c=c))/h
	J[2,2] = imag(g(u,v+im*h;a=a,c=c))/h
	
	return J
end

function stability(J)
	return maximum(real(eigvals(J)))
end

function stabilityplot(;N=10000)

	trJ = zeros(Float64,N)
	detJ = zeros(Float64, N)
	lJ = zeros(Float64, N)
	for n in 1:N
		J = randn(Float64, 2, 2)
		lJ[n] = stability(J)
		trJ[n] = tr(J)
		detJ[n] = det(J)
	end
	plt.scatter(trJ, detJ, c=lJ, s=1.0, cmap="bwr", vmin=-1, vmax=+1)
	plt.colorbar(label="\$ \\max \\, \\Re (\\lambda) \$", extend="both")
	plt.plot(-5.0:0.1:5.0, ((-5.0:0.1:5.0).^2.0)./4.0, "-k", label="\$ \\Delta = 0 \$")
	plt.xlabel("tr\$ A \$")
	plt.ylabel("\$ \\det A \$")
	plt.xlim([-5,5])
	plt.ylim([-5,5])
	plt.legend(loc=0, edgecolor="none")
end

function oscillationplot(;N=10000)

	trJ = zeros(Float64,N)
	detJ = zeros(Float64, N)
	lJ = zeros(Float64, N)
	for n in 1:N
		J = randn(Float64, 2, 2)
		lJ[n] = maximum(abs.(imag.(eigvals(J))))
		trJ[n] = tr(J)
		detJ[n] = det(J)
	end
	plt.scatter(trJ, detJ, c=lJ, s=1.0, cmap="bwr", vmin=-1, vmax=+1)
	plt.colorbar(label="\$ \\max \\, \\Im (\\lambda) \$", extend="both")
	plt.plot(-5.0:0.1:5.0, ((-5.0:0.1:5.0).^2.0)./4.0, "-k", label="\$ \\Delta = 0 \$")
	plt.xlabel("tr\$ A \$")
	plt.ylabel("\$ \\det A \$")
	plt.xlim([-5,5])
	plt.ylim([-5,5])
	plt.legend(loc=0, edgecolor="none")
end

function FHNstrmplt(; a=0.37, b=0.1, c=0.05, i=0.0)
	U,V,F,G = FHNgrid(; a=a, b=b, c=c, i=i)
	
	fig = plt.streamplot(U,V,F,G, color=sqrt.(F.^2 + G.^2))
	plt.colorbar(label="\$ \\sqrt{\\dot u ^2 + \\dot v ^2} \$")
	plt.xlabel("\$ u \$")
	plt.ylabel("\$ v \$")
	plt.xlim([-2,2])
	plt.ylim([-1,1])
	return fig
end

function FHNtotal(; a=0.37, b=0.1, c=0.05, i=0.0)
	fig = plt.figure()
	FHNstrmplt(;a=a,b=b,c=c,i=i)
	FHNnullclines(;a=a,b=b,c=c,i=i)
	r = FHNequilibria(;a=a,b=b,c=c,i=i)
	for n in 1:size(r,1)
		u = r[n,1]; v = r[n,2]
		l = eigvals(FHNjac(u,v;a=a,b=b,c=c,i=i))
		ii = argmax(real(l))
		if maximum(real(l)) > 0
			plt.plot([u],[v],"ok", markerfacecolor="w", label="\$ \\lambda = $(round(l[ii],sigdigits=2)) \$")
		elseif maximum(real(l)) < 0
			plt.plot([u],[v],"ok", label="\$ \\lambda = $(round(l[ii],sigdigits=2)) \$")
		end
	end
	plt.legend(loc=0, edgecolor="none")
	return fig
end

function FHNdyn(; a=0.37, b=0.1, c=0.05, i=0.0, x0=randn(Float64,2))
	function fhn!(dx, x, p, t)
		dx[1] = f(x[1],x[2];b=p[2],i=p[4])
		dx[2] = g(x[1],x[2];a=p[1],c=p[3])
		return nothing
	end
	prob = ODEProblem(fhn!, x0, (0.0, 1000.0), [a,b,c,i])
	sol = solve(prob, abstol=1e-13, reltol=1e-13)
	
	FHNtotal(;a=a,b=b,c=c,i=i)
	plt.plot(sol[1,:], sol[2,:], "-k", linewidth=3)
	plt.plot(x0[1],x0[2],"xk")
	plt.title("\$ \\alpha=$(a), \\beta=$(b), \\gamma=$(c), I=$(round(i,sigdigits=3)) \$")
	plt.xlabel("\$ u \$")
	plt.ylabel("\$ v \$")
	plt.xlim([-.5,1.5])
	plt.ylim([-.5,.5])
	return nothing

end

function FHNexc(; a=0.37, b=0.1, c=0.05, i=0.0)

	U,V,F,G = FHNgrid(; a=a, b=b, c=c, i=i, ulims=[-0.3,1.3], vlims=[-0.3,0.3], NU=129, NV=65)
	
	fig, axs = plt.subplots(3,4,sharex=true, sharey=true, constrained_layout=true)
	for (n,u) in enumerate(range(-0.1,0.2,step=0.1)), (m,v) in enumerate(range(0.05,-0.05,step=-0.05))
		axs[m,n].streamplot(U,V,F,G,color=sqrt.(F.^2 + G.^2),linewidth=sqrt.(F.^2 + G.^2))
		axs[m,n].contour(U,V,F,levels=[-1e-14,+1e-14],colors="C0",linestyles="solid")
		axs[m,n].contour(U,V,G,levels=[-1e-14,+1e-14],colors="C1",linestyles="solid")
		
		function fhn!(dx, x, p, t)
			dx[1] = f(x[1],x[2];b=p[2],i=p[4])
			dx[2] = g(x[1],x[2];a=p[1],c=p[3])
			return nothing
		end
		prob = ODEProblem(fhn!, [u;v], (0.0, 1000.0), [a,b,c,i])
		sol = solve(prob, abstol=1e-13, reltol=1e-13)
		
		axs[m,n].plot(sol[1,:], sol[2,:], "-k", linewidth=2)
		axs[m,n].plot(u,v,"xk")
		axs[m,n].plot(0,0,".k")

		
		if n==1
			axs[m,n].set_ylabel("\$ v_0 = $(v) \$")
		end
		if m==3
			axs[m,n].set_xlabel("\$ u_0 = $(u) \$")
		end
		axs[m,n].set_xlim([-0.3,1.3])
		axs[m,n].set_ylim([-0.3,0.3])
	end
		
	return fig, axs

end

function currentBif(;a=0.37,b=0.1,c=0.05,i=0.0)
	ii = range(0.00,0.05,step=0.0001)
	li = zeros(Float64, length(ii))
	for (n,i) in enumerate(ii)
		x0 = FHNequilibria(;i=i)
		x0 = x0[argmin(x0[:,1]),:]
		li[n] = stability(FHNjac(x0[1],x0[2];i=i))
	end
	fig = figure()
	plt.plot(ii, li, "-", label="\$ \\max \\Re( \\lambda)(I) \$")
	ind = argmin(abs.(li))
	plt.plot(ii[ind], li[ind], ".k", label="\$ I^* = $(ii[ind]) \$")
	plt.xlabel("\$ I \$")
	plt.ylabel("")
	plt.legend(loc=0,edgecolor="none")
	return fig
end

function currentBifTraj(;a=0.37,b=0.1,c=0.05,i=0.0)
	function fhn!(dx, x, p, t)
		dx[1] = f(x[1],x[2];b=p[2],i=p[4])
		dx[2] = g(x[1],x[2];a=p[1],c=p[3])
		return nothing
	end
	fig, axs = plt.subplots(5,1,sharex=true,sharey=true,constrained_layout=true)
	
	for (n,i) in enumerate(range(0.0,0.04,length=5))
		prob = ODEProblem(fhn!, [1.0;-0.1], (0.0, 1000.0), [a,b,c,i])
		sol = solve(prob, abstol=1e-13, reltol=1e-13)
		axs[n].plot(sol.t, sol[1,:], "-C0", label="\$ u(t) \$")
		axs[n].plot(sol.t, sol[2,:], "-C1", label="\$ v(t) \$")
		axs[n].set_ylabel("\$ I = $(round(i,sigdigits=2)) \$")
	end
	axs[1].legend(loc=0, edgecolor="none")
	axs[end].set_xlabel("\$ t \$")
	axs[end].set_xlim([0,200])
	plt.savefig("/Users/christophermarcotte/Downloads/fhncurrentbiftraj.pdf")
	plt.close("all")
end

function alphaBifPlot(;a=0.37,b=0.1,c=0.05,i=0.0)
	fig = plt.figure()
	astar = 0.2025
	for a in range(0.37, -0.07, length=4096)
		rr = FHNequilibria(;a=a,b=b,c=c,i=i)
		for n in 1:size(rr,1)
			J = FHNjac(rr[n,1],rr[n,2]; a=a,b=b,c=c,i=i)
			l = eigvals(J)
			if maximum(real(l)) > 0
				plt.plot(a,rr[n,1],".k")
			elseif maximum(real(l)) < 0
				plt.plot(a, rr[n,1],"ok")
			end
		end			
	end
	plt.xlabel("\$ \\alpha \$")
	plt.ylabel("\$ \\bar{u} \$")
	plt.title("Thick - Stable; Thin - Unstable")
	return fig
end

function FHNLC(; a=0.37, b=0.1, c=0.05, i=0.04)
	FHNtotal(;a=a,b=b,c=c,i=i)
	function fhn!(dx, x, p, t)
		dx[1] = f(x[1],x[2];b=p[2],i=p[4])
		dx[2] = g(x[1],x[2];a=p[1],c=p[3])
		return nothing
	end
	prob = ODEProblem(fhn!, [0.0;-0.1], (0.0, 1000.0), [a,b,c,i])
	sol = solve(prob, abstol=1e-13, reltol=1e-13)
	plt.plot(sol[1,:], sol[2,:], "-k", linewidth=2)
	plt.plot(sol[1,1], sol[2,1],"xk")
	
	prob = ODEProblem(fhn!, [0.0;0.1], (0.0, 1000.0), [a,b,c,i])
	sol = solve(prob, abstol=1e-13, reltol=1e-13)
	plt.plot(sol[1,:], sol[2,:], "-k", linewidth=2)
	plt.plot(sol[1,1], sol[2,1],"xk")
	
	plt.title("\$ \\alpha=$(a), \\beta=$(b), \\gamma=$(c), I=$(round(i,sigdigits=3)) \$")
	plt.xlabel("\$ u \$")
	plt.ylabel("\$ v \$")
	plt.xlim([-.5,1.5])
	plt.ylim([-.2,.3])

	plt.savefig("/Users/christophermarcotte/Downloads/fhnLC.pdf")
	plt.close("all")
end
