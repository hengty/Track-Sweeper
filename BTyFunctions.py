#ty@wisc.edu
#Last update: 08 Jan 2022

import os
import numpy as np
from functools import reduce
import scipy.optimize, scipy.signal
from icecube import icetray, dataio, dataclasses, simclasses, recclasses, MuonGun
import healpy as hp

#convert (r,theta) to (x,y)
def pol2rec(polar):
	return polar[0]*np.array([np.cos(np.deg2rad(polar[1])), np.sin(np.deg2rad(polar[1]))])

#take in polygon's vertices and return the polygon's (signed) area
def polygon_area(v):   #polygon vertices in (x,y) coordinates
	#Shoelace formula for calculating (oriented) area of a simple polygon
	numsides = len(v)
	area = 0.0
	for i in range(0, numsides):
		j = (i+1)%numsides
		area = area + (v[i][0]*v[j][1] - v[j][0]*v[i][1])
	return 0.5*area
	
#compute the centroid (geometric center) of a simple polygon
#input: vertices of polygon in cartesian
def polygon_centroid(v):
	numsides = len(v)
	if(numsides==0): return np.array([])
	centroid = np.zeros(2)
	for i in range(0, numsides):
		j = (i+1)%numsides
		centroid = centroid + np.array([v[i][0]+v[j][0], v[i][1]+v[j][1]])*(v[i][0]*v[j][1] - v[j][0]*v[i][1])
	return centroid/(6*polygon_area(v))
	
#input: time ordered list of strings in the neighborhood
#output: numpy array representing the polygon resulting from the string ordering
#input stringlist must be of strings from only one neighborhood
def polygon_generator(stringlist, numsub, neighbor_map, neighborhood):
	if(numsub>=len(stringlist)): raise ValueError('second argument can not be larger than length of first argument')
	#start with the closest-string's territory, in cartesian coordinates
	curnt_territory = [pol2rec(vertex) for vertex in neighborhood['territory']]
	
	full_stringset = set([stringlist[0]] + list(neighbor_map[:,0].astype(int)))
	missing_strings = full_stringset - set(stringlist)
	stringlist = list(stringlist) + list(missing_strings)   #ensure the presence of all strings in the neighborhood
	
	#get subregion from specified subsequent hit strings
	for sub in range(1, numsub+1):
		curnt_ngbr = pol2rec(neighborhood[stringlist[sub]])
		for cut in range(sub+1, len(stringlist)):
			stringlist_consistency = False
			curnt_cut = pol2rec(neighborhood[stringlist[cut]])
			#cutter defines boundary between sub string and curnt_ngbr string
			cutter = [0.5*(curnt_ngbr+curnt_cut)]   #midway point between curnt_ngbr and curnt_cut
			cutter.append([-(curnt_cut[1]-cutter[0][1])+cutter[0][0], 
						curnt_cut[0]-cutter[0][0]+cutter[0][1]])   #(curnt_cut-cutter[0] rotated by 90 degrees)
			#u is a "rotation" that align cutter with the x axis
			c = cutter[1]-cutter[0]
			u = np.array([[c[0], c[1]], [-c[1], c[0]]])   #(need not be norm 1)
			cutterprime = np.array([[np.sum(u[0]*cutter[0]), np.sum(u[1]*cutter[0])], 
									[np.sum(u[0]*cutter[1]), np.sum(u[1]*cutter[1])]])
			#check cutterprime alignment with positive x-axis
			if(not np.isclose(cutterprime[0][1], cutterprime[1][1]) or cutterprime[0][0]>=cutterprime[1][0]): 
				raise ValueError('cutter rotation failed')
			#check curnt_ngbr above or below cutter prime
			if(np.sum(u[1]*curnt_ngbr) > cutterprime[0][1]): curnt_ngbr_above = True
			else: curnt_ngbr_above = False
			
			#Once curnt_territory and cutter are determined, find the intersections
			cutpoints = []
			#go through the line segments that define curnt_territory, one by one
			for i in range(len(curnt_territory)):
				#rotate by u, and y shift
				segment = [curnt_territory[i], curnt_territory[(i+1)%len(curnt_territory)]]
				segprime= np.array([[np.sum(u[0]*segment[0]), np.sum(u[1]*segment[0])-cutterprime[0][1]],
									[np.sum(u[0]*segment[1]), np.sum(u[1]*segment[1])-cutterprime[0][1]]])
				#if cutter intersects with the segment, save the segment point that is on the same side as curnt_ngbr
				# as well as the distance from segment[0] to the intersection point
				if(np.sign(segprime[0][1])!=np.sign(segprime[1][1])):   #check intersection
					if(curnt_ngbr_above): side2save = np.argmax(segprime[:,1])   #make sure to identify the point on the
					else: side2save = np.argmin(segprime[:,1])                   #same "side" as the curnt_ngbr
					#calculate and save distance ratio from segment[0] to the intersection point
					r = (segment[1]-segment[0])*segprime[0][1]/(segprime[0][1]-segprime[1][1])
					if(np.linalg.norm(r)!=0): cutpoints.append([i, r, side2save])   #conditional here in case of intersection at a vertex
				else:    #if no intersection try to prove stringlist is possible
					if(np.sign(segprime[0][1])==1 and curnt_ngbr_above): stringlist_consistency = True
					if(np.sign(segprime[0][1])==-1 and not curnt_ngbr_above): stringlist_consistency = True
						
			#cut, if collected exactly two intersections. else, no cut
			if(len(cutpoints)==2):
				if(cutpoints[0][2]==cutpoints[1][2]): raise ValueError('Conflicting cutpoints')
				if(cutpoints[0][2]==0): cutpoints = cutpoints[::-1]   #make sure the first point in cutpoints goes counterclockwise
				new_territory = [curnt_territory[cutpoints[0][0]]+cutpoints[0][1]]   #first element of new_territory is the first point in cutpoints
				new_range = [cutpoints[0][0], cutpoints[1][0]]
				if(new_range[1]<new_range[0]): new_range[1] = new_range[1] + len(curnt_territory)   #modulo later
				for new_index in range(new_range[0]+1, new_range[1]+1):
					new_territory.append(curnt_territory[new_index%len(curnt_territory)])
				new_territory.append(curnt_territory[cutpoints[1][0]]+cutpoints[1][1])   #last element of new_territory
				curnt_territory = new_territory[:]   #true copy
			#if no cut, still make sure the curnt_territory is closer to curnt_ngbr than curnt_cut
			else:
				if(not stringlist_consistency): return []
	return curnt_territory
	
def plot_neighborhood(neighbor_map, MSS):
	neighbors = pol2rec(neighbor_map[:,1:].T)
	plt.plot([0],[0],'ko')
	plt.annotate(str(MSS), xy=(0,0))
	plt.plot(neighbors[0], neighbors[1], 'o')
	for i in range(len(neighbor_map[:,0])):
		plt.annotate(str(int(neighbor_map[:,0][i])), xy=(neighbors[0][i],neighbors[1][i]))
	plt.ylim(-140,140)
	plt.xlim(-140,140)
	plt.xlabel('x (m)')
	plt.ylabel('y (m)')
	return
	
#From the GCD file, generate the territory maps of IceCube
def maps_generator(GCD_filename):
	geo = dataio.I3File(GCD_filename, 'r')
	g = geo.pop_frame()
	while(str(g.Stop) != 'Geometry'): g = geo.pop_frame()
	near_defn = 160.
	
	neighbor_map = [np.array([[-1,60,0],[-2,60,180]])]
	for string in range(1, 79):
		neighbors = []
		possible_ngbr = list(range(1,79))
		possible_ngbr.remove(string)
		for neighbor in possible_ngbr:
			str2ngbr = g['I3Geometry'].omgeo[icetray.OMKey(neighbor,1,0)].position - g['I3Geometry'].omgeo[icetray.OMKey(string,1,0)].position
			if(str2ngbr.rho<near_defn):
				neighbors.append(np.array([neighbor, str2ngbr.rho, np.rad2deg(str2ngbr.phi)]))
		neighbor_map.append(np.array(neighbors))
    
	#describe string's territory with the circumcenters and halfway points - a list of dictionaries
	territory_map = []
	for string in neighbor_map:
		string = string[string[:,2].argsort()]   #make sure the neighbor strings are sorted smallest to largest polar angles
		stringdict = {}
		territory = []
		for i in range(string.shape[0]):
			if(i < string.shape[0]-1):
				php = [string[i][1]/2., string[i+1][1]/2.]
				thetahp = np.deg2rad([string[i][2], string[i+1][2]])
			else:
				php = [string[i][1]/2., string[0][1]/2.]
				thetahp = np.deg2rad([string[i][2], string[0][2]+360])
			func = lambda theta : np.cos(theta) - (php[0]/php[1])*np.cos(thetahp[1]-thetahp[0]-theta)
			theta_guess = (thetahp[1]-thetahp[0])*0.5
			theta = scipy.optimize.fsolve(func, theta_guess)   #find root to get the circumcenter
			stringdict[int(string[i][0])] = string[i][1:]
			territory.append([php[0]/np.cos(theta[0]), np.rad2deg(theta[0]+thetahp[0])])
		stringdict['territory'] = territory
		territory_map.append(stringdict)

	boundary_strings = [1,2,3,4,5,6,13,21,30,40,50,59,67,74,73,72,78,77,76,75,68,60,51,41,31,22,14,7]
	deep_strings = [79, 80, 81, 82, 83, 84, 85, 86]
	
	#Boundary DOMs
	boundaries = {'top':np.stack((np.arange(1,79), np.repeat(1,78))).T,
					'bottom':np.stack((np.arange(1,79), np.repeat(60,78))).T,
					'side1':np.stack((np.repeat([1,2,3,4,5,6], 60), np.tile(np.arange(1,61), 6))).T,
					'side2':np.stack((np.repeat([6,13,21,30,40,50], 60), np.tile(np.arange(1,61), 6))).T,
					'side3':np.stack((np.repeat([50,59,67,74], 60), np.tile(np.arange(1,61), 4))).T,
					'side4':np.stack((np.repeat([74,73,72,78,77,76,75], 60), np.tile(np.arange(1,61), 7))).T,
					'side5':np.stack((np.repeat([75,68,60,51,41,31], 60), np.tile(np.arange(1,61), 6))).T,
					'side6':np.stack((np.repeat([31,22,14,7,1], 60), np.tile(np.arange(1,61), 5))).T}
	boundaries_all = np.full((79,61), False)
	for side in boundaries: boundaries_all[tuple(boundaries[side].T)] = True
	boundaries['all'] = np.argwhere(boundaries_all)   #all of the boundary DOMs
			
	#DOMcoord -- IceCube coordinate of all DOMs
	#with this can get zmap=DOMcoord[:,:,2]
	DOMcoord = np.full((79,61,3),np.nan)
	DOMcoord[1:,0] = np.inf
	for string in range(1,79):
		for layer in range(1,61):
			DOMcoord[string,layer] = np.array(g['I3Geometry'].omgeo[icetray.OMKey(string,layer,0)].position)
	
	#Detector bounds
	xybounds = DOMcoord[[1,6,50,74, 78,75,31], 1][:,0:2]
	ztop = np.average(DOMcoord[tuple(boundaries['top'].T)][:,2])
	zbottom = np.average(DOMcoord[tuple(boundaries['bottom'].T)][:,2])
	zbounds = np.array([ztop, zbottom])

	geo.close()
	return neighbor_map, territory_map, boundary_strings, boundaries, deep_strings, g['I3Geometry'], DOMcoord, xybounds, zbounds
	
#convert IceCube layer to IceCube z coordinate
def layer2z(layer, I3Geometry, signf_string):
	zfloor = int(layer)
	zratio = layer - zfloor
	if(zfloor>=0 and zfloor<60):
		return (1-zratio)*I3Geometry.omgeo[icetray.OMKey(signf_string,zfloor,0)].position[2] + \
		          zratio*I3Geometry.omgeo[icetray.OMKey(signf_string,zfloor+1,0)].position[2]
	else: return np.nan
	
#determine location and direction of a simulated cascade by
#the position and direction of the most energetic electron in the shower
def shower_max(tree, rho=600., z=600.):
    max_energy = 0
    max_time = np.nan
    max_pos = np.nan
    for particle in tree:
        if(np.isnan(particle.length) or np.isclose(particle.length, 0.0)): continue
        if(not str(particle.type) in ['EMinus', 'EPlus', 'MuMinus', 'MuPlus', 'TauMinus', 'TauPlus']):
            if(str(particle.type)[0:2]!='Nu'): print("Warning: ignoring "+str(particle.type))
            continue
        if(particle.pos.rho<rho and -z<particle.pos.z<z and particle.energy>max_energy):
            max_time = particle.time
            max_pos = particle.pos
            max_energy = particle.energy
    return max_time, max_pos, max_energy
	
#Estimate deposited energy in the cascade, by substracting out the energy of the escaping neutrino
def shower_energy(tree, interaction_bound=100):
	shower_max_pos = shower_max(tree)[1]
	shower_head = [None, 0]
	#the most energetic particle in the shower
	for particle in tree:
		dist2shower = np.linalg.norm(np.array(particle.pos-shower_max_pos))
		if(dist2shower<interaction_bound and shower_head[1]<particle.energy):
			shower_head = [particle, particle.energy]
	#the neutrino that causes the shower is the parent of the most energetic particle in the shower
	nu_daughters = tree.get_daughters(tree.parent(shower_head[0].id))
	return sum([particle.energy for particle in nu_daughters if(str(particle.type)[:2]!='Nu')])
	
#linear interpolate to return value in y specified by input point in x
#x and y must be numpy array, x strictly increasing
def linear_read(point, x, y):
	if(point<=x[0]): return y[0]
	if(point>=x[-1]): return y[-1]  
	left = np.sum(x<=point)-1
	weight = (point-x[left])/(x[left+1]-x[left])
	return y[left]*(1-weight) + y[left+1]*weight

#take in raw cumulative pulse series (time and charge), return its "width" and left edge used to calculate the "width"
#"width" defined here as the smallest time interval to capture a certain fraction of the total PEs of the pulse
def pulse_series_width(pulse_series, width_defn=0.5, cut_param=1000):
	times, charging = pulse_series[:,0], pulse_series[:,1]
	midtime = linear_read(0.5*charging[-1], charging, times)   #midtime is where the meat of the signal is
	cutend = np.sum(times<=midtime+cut_param)     #for cutting out end noises
	cutbegin=np.sum(times<=midtime-cut_param)   #for cutting out beginning noises
	chargecut = 0. if cutbegin==0 else charging[cutbegin-1]
	times = times[cutbegin:cutend]                              #cut
	charging = charging[cutbegin:cutend]/(charging[cutend-1]-chargecut)   #cut and normalize
	if(len(times)==0): return np.inf, np.inf, midtime   #call over if nothing left after cut
	timelimit = linear_read(1.0-width_defn, charging, times)   #upper limit of search box
	width_min = np.inf
	left_edge_min = np.inf
	for left_edge in np.arange(times[0], timelimit):   #search in increment of 1ns
		charge_frac = linear_read(left_edge, times, charging) + width_defn
		width = linear_read(charge_frac, charging, times) - left_edge
		if(width<width_min):
			width_min = width
			left_edge_min = left_edge
	return width_min, left_edge_min, midtime
	

#Use time-of-arrival of 4 DOMs to locate cascade's (x,y,z)
#input: (t,x,y,z) of 4 DOMs -- a 4x4 numpy array, initial guess (IceCube coordinates)
#
#output: guesses cascade's (t,x,y,z)
def multilateration(events, guess, speed=0.222, spread=0.01, numtrials=100):
	t0, r0 = events[0][0], events[0][1:]
	t1, r1 = events[1][0], events[1][1:]
	t2, r2 = events[2][0], events[2][1:]
	t3, r3 = events[3][0], events[3][1:]
	constraints = lambda r: [np.linalg.norm(r0-r[1:])-v0*(t0-r[0]),
							np.linalg.norm(r1-r[1:])-v1*(t1-r[0]),
							np.linalg.norm(r2-r[1:])-v2*(t2-r[0]),
							np.linalg.norm(r3-r[1:])-v3*(t3-r[0])]
	emitter=[]
	for trial in range(numtrials):
		v0,v1,v2,v3 = speed + spread*(np.random.rand(4)*2 - 1)   #0.222 m/ns expected photon velocity
		emitter.append(scipy.optimize.fsolve(constraints, guess))
	return np.array(emitter)
def trilateration(events, guess, speed=0.222, spread=0.01, numtrials=1000):
	t0, r0 = events[0][0], events[0][1:]
	t1, r1 = events[1][0], events[1][1:]
	t2, r2 = events[2][0], events[2][1:]
	constraints = lambda r: [np.linalg.norm(r0-r[1:])-v0*(t0-r[0]),
							np.linalg.norm(r1-r[1:])-v1*(t1-r[0]),
							np.linalg.norm(r2-r[1:])-v2*(t2-r[0])]
	emitter=[]
	for trial in range(numtrials):
		v0,v1,v2 = speed + spread*(np.random.rand(3)*2 - 1)   #0.222 m/ns expected photon velocity
		solution = scipy.optimize.fsolve(constraints, guess)
		check = np.isclose(max(constraints(solution)),0.0)
		if(check): emitter.append(solution)
	return np.array(emitter)
	
#lateration using DOMs from a single string
#r is the hyponthenus of a right triangle with legs z and l
#r[t, z, l]
def zlateration(events, guess, speed=0.222, spread=0.01, numtrials=100, bounds=np.array([[9000,10000],[-500,500],[0,70]])):
    t0, r0 = events[0][0], np.array([events[0][1], 0.])
    t1, r1 = events[1][0], np.array([events[1][1], 0.])
    t2, r2 = events[2][0], np.array([events[2][1], 0.])
    constraints = lambda r: [np.linalg.norm(r0-r[1:])-v0*(t0-r[0]),
                            np.linalg.norm(r1-r[1:])-v1*(t1-r[0]),
                            np.linalg.norm(r2-r[1:])-v2*(t2-r[0])]
    emitter=[]
    for trial in range(numtrials):
        v0,v1,v2 = speed + spread*(np.random.rand(3)*2 - 1)   #0.222 m/ns expected photon velocity
        solution = np.array(scipy.optimize.fsolve(constraints, guess))
        if(np.all(solution>bounds[:,0]) and np.all(solution<bounds[:,1])): emitter.append(solution)
    return emitter


#given a cumulative pulse series, return the best fstHit and 1st,2nd,3rd derivatives at fstHit
def rising_edge_fitter(curve, width_defn=0.5, cut_param=1000):
	if(len(curve)==0): return [np.inf]*4, [np.inf]*4
	width, left_edge, midtime = pulse_series_width(curve, width_defn=width_defn, cut_param=cut_param)
	right_edge = left_edge+width
	fgp = (left_edge<curve[:,0])*(curve[:,0]<right_edge)
	if(np.sum(fgp)<2 or right_edge==np.inf): return [np.inf]*4, [np.inf]*4     #in case pulse_series_width() failed or width too small
	rightval = linear_read(right_edge, curve[:,0], curve[:,1])
	times = list(curve[:,0][fgp])+[right_edge]   #include the bounds (for smaller signals)
	charges = list(curve[:,1][fgp])+[rightval]      #include the bounds (for smaller signals)
	bounds = [[left_edge-width, 0, -rightval, -rightval], 
			[left_edge, rightval, rightval, 0]]
	poly_app = lambda t, t0, c1, c2, c3: c1*(t-t0) + c2*(t-t0)**2 + c3*(t-t0)**3
	try: fit, pcov = scipy.optimize.curve_fit(poly_app, times, charges, bounds=bounds)   #might want to change this to least abs instead of square
	except: return [left_edge]+[np.nan]*3, [np.nan]*4
	return fit, np.sqrt(np.diag(pcov))   #[fstHit] + [three polynomial coefficients]
#rising_edge_fitter, but only up to 2nd derivatives
def rising_edge_fitter2(curve, width_defn=0.5, cut_param=1000):
	if(len(curve)==0): return [np.inf]*3, [np.inf]*3
	width, left_edge, midtime = pulse_series_width(curve, width_defn=width_defn, cut_param=cut_param)
	right_edge = left_edge+width
	fgp = (left_edge<curve[:,0])*(curve[:,0]<right_edge)
	if(np.sum(fgp)<2 or right_edge==np.inf): return [np.inf]*3, [np.inf]*3     #in case pulse_series_width() failed or width too small
	rightval = linear_read(right_edge, curve[:,0], curve[:,1])
	times = list(curve[:,0][fgp])+[right_edge]   #include the bounds (for smaller signals)
	charges = list(curve[:,1][fgp])+[rightval]   #include the bounds (for smaller signals)
	bounds = [[left_edge-width, 0, -rightval], 
			  [left_edge, rightval, 0]]
	poly_app = lambda t, t0, c1, c2: c1*(t-t0) + c2*(t-t0)**2
	try: fit, pcov = scipy.optimize.curve_fit(poly_app, times, charges, bounds=bounds)   #might want to change this to least abs instead of square
	except: return [left_edge]+[np.nan]*2, [np.nan]*3
	return fit, np.sqrt(np.diag(pcov))   #[fstHit] + [three polynomial coefficients]

	
###Depreciated
#given cumulative pulse series, return best fit for the signal start time
#best fit is root of the polynomial fit to the fastest growing part of waveform
#only return the best fit if root is found sucessfully and the value is later than first data and left_edge
#return best fit and the polynomial coefficients
def first_hit(waveform, fitpower=2, width_defn=0.5):
	width, left_edge, midtime = pulse_series_width(waveform, width_defn=width_defn, cut_param=1000)
	right_edge = left_edge+width
	if(left_edge==right_edge): return np.inf, [np.inf]   #check if pulse_series_width() failed
	fgp = (waveform[:,0]>left_edge)*(waveform[:,0]<right_edge)   #fastest growing part
	leftvalue = linear_read(left_edge, waveform[:,0], waveform[:,1])
	rightvalue = linear_read(right_edge, waveform[:,0], waveform[:,1])
	if(np.sum(fgp)!=0):
		times = [left_edge]+list(waveform[:,0][fgp])+[left_edge+width]   #include the bounds (for smaller signals)
		charges = [leftvalue]+list(waveform[:,1][fgp])+[rightvalue]      #include the bounds (for smaller signals)
		for n in range(fitpower, 0, -1):
			fit = np.polyfit(times, charges, n)
			polynomial = lambda t : np.polyval(fit, t)
			root = scipy.optimize.fsolve(polynomial, left_edge)[0]
			if(waveform[0][0]<=root<=left_edge and np.isclose(polynomial(root),0.0)): return root, fit
			
	#linear fit, if it gets to this point
	#slope = (rightvalue-leftvalue)/width
	#b = leftvalue-slope*left_edge
	return left_edge, [np.inf]
	

###Depreciated
#given cumulative pulse series, return best guess of time of first non noise hit
#return np.inf if can't find hit satisfying non noise condition:
#requires four "nicely aligned" hits: the three slopes are larger than threshold
def start_time(waveform, slope_thres=10**-3, deltaT_thres=1):
	width, left_edge, midtime = pulse_series_width(waveform, width_defn=0.5, cut_param=500)
	deltatimes, deltacharges = np.diff(waveform[:,0]), np.diff(waveform[:,1])
	slopes = deltacharges/deltatimes
	minm_slope = slope_thres*0.5*waveform[:,1][-1]/width   #a tiny fraction of the characteristic slope
	for n in range(len(slopes)-3):   #effectively ignoring waveform of 4 hits or less
		if(np.max(deltatimes[n:n+3])>deltaT_thres*width): continue
		if(np.min(slopes[n:n+3])>minm_slope): return waveform[:,0][n]
	return np.inf

###Depreciated
#given cumulative pulse series, return interval estimating start time
#by drawing lines through early data points to intersect the x axis
#width_defn (0.0,1.0) controls how wide the interval is
def start_time_old(waveform, width_defn=0.2):
	if(len(waveform)<6): return np.inf, np.inf
	times, charges = waveform[:,0], np.log(waveform[:,1])
	mid_point = np.sum(charges <= charges[-1]/2.) #use points up to half of total charges
	zeroes = []
	for n in range(mid_point):
		slope = (charges[n+1]-charges[n])/(times[n+1]-times[n])
		zeroes.append((-charges[n]+slope*times[n])/slope)
	zeroes = np.sort(zeroes)   #sort to find densest part
	counts = np.array(range(1,len(zeroes)+1))
	#search for the densest part in zeroes
	#search interval starts from zeroes[0] to timelimit
	width_defn = width_defn*counts[-1]
	timelimit = linear_read(counts[-1]-width_defn, counts, zeroes)   #upper limit of search interval
	width_min = np.inf
	left_edge_min = np.inf
	for left_edge in np.arange(zeroes[0], timelimit):   #search in increment of 1ns
		right_counts = linear_read(left_edge, zeroes, counts) + width_defn
		width = linear_read(right_counts, counts, zeroes) - left_edge
		if(width<width_min):
			width_min = width
			left_edge_min = left_edge
	return left_edge_min, width_min
	
	
#takes in a DOM number, DOM depths, DOM values of entire string, number of DOMs to fit,and polynomial power to fit to
#return extremum from the polynomial fit
def zfitter(midDOM, DOMz, stringdata, numDOMs, power):
	DOMs2fit = [midDOM]
	numDOMsleft = numDOMs-1
	level=1
	while(numDOMsleft>0 and level<10):   #the 10 is kind of arbitrary
		if(midDOM-level>0):
			if(stringdata[midDOM-level]!=np.inf):
				DOMs2fit.append(midDOM-level)
				numDOMsleft-=1
			else: print('Warning: missing DOM'+str(midDOM-level))
		if(midDOM+level<61):
			if(stringdata[midDOM+level]!=np.inf):
				DOMs2fit.append(midDOM+level)
				numDOMsleft-=1
			else: print('Warning: missing DOM'+str(midDOM+level))
		level+=1
	if(len(DOMs2fit)<power+1): return np.nan,np.nan	
	DOMs2fit.sort()
	fit_coef=np.polyfit(DOMz[DOMs2fit], stringdata[DOMs2fit], power)
	x = np.arange(DOMz[midDOM]-40, DOMz[midDOM]+40, 0.1)   #0.1m
	y = np.polyval(fit_coef, x)
	rough_guess = np.polyval(fit_coef, DOMz[midDOM])
	if(rough_guess<y[0] and rough_guess<y[-1]): return x[np.argmin(y)],fit_coef
	if(rough_guess>y[0] and rough_guess>y[-1]): return x[np.argmax(y)],fit_coef
	return np.nan,fit_coef

#special case of zfitter() -- quadratic fit
#takes in a DOM number, DOM depths, DOM values of entire string, number of DOMs to fit
#return the vertex of the quadratic fit if a good fit
#else just return midDOM's z coordinate
def zfitter2(midDOM, DOMzs, stringdata, numDOMs):
	DOMs2fit = [midDOM]
	numDOMsleft = numDOMs-1
	level=1   #try to build list of DOMs to fit, using DOMs closest as possible to midDOM
	while(numDOMsleft>0 and level<10):   #the 10 is kind of arbitrary
		if(midDOM-level>0):
			if(stringdata[midDOM-level]!=np.inf):
				DOMs2fit.append(midDOM-level)
				numDOMsleft-=1
			#else: print 'Warning: missing DOM'+str(midDOM-level)
		if(midDOM+level<61):
			if(stringdata[midDOM+level]!=np.inf):
				DOMs2fit.append(midDOM+level)
				numDOMsleft-=1
			#else: print 'Warning: missing DOM'+str(midDOM+level)
		level+=1
	if(len(DOMs2fit)<3): return DOMzs[midDOM],stringdata[midDOM]   #just use midDOM if can't find enough number of DOMs
	DOMs2fit.sort()
	a,b,c=np.polyfit(DOMzs[DOMs2fit], stringdata[DOMs2fit], 2)
	z_fit = -b/(2*a)
	value_fit = c-b*b/(4*a)
	return z_fit, value_fit
		
#input: I3MCTree
#output: plot of visible energy the particle shower
def plot_shower(tree):
	primary = tree.get_head()
	dir_vector = np.array([primary.dir.x, primary.dir.y, primary.dir.z])
	origin = primary.pos
	tracklist = []   #of visible leptons
	max_energy = 0
	max_pos = np.nan
	for particle in tree:
		if(np.isnan(particle.length) or np.isclose(particle.length, 0.0)): continue
		if(not str(particle.type) in ['EMinus', 'EPlus', 'MuMinus', 'MuPlus']):
			if(str(particle.type)[0:2]!='Nu'): print("Warning: ignoring "+str(particle.type))
			continue
		pos_prime = particle.pos-origin
		if(not np.isclose(np.dot(pos_prime, dir_vector), pos_prime.r)): raise ValueError('One-track assumption broken')
		dist = pos_prime.r   #distance from vertex to particle
		tracklist.append([dist, dist+particle.length, particle.energy])
		if(particle.pos.rho<600.0 and -600.0<particle.pos.z<600.0 and particle.energy>max_energy):
			max_pos = particle.pos
			max_energy = particle.energy
	tracklist = np.array(tracklist)
	dist2max = (max_pos-origin).r
	
	#build path segments and visible energy along the path
	segments = np.sort(list(set(tracklist[:,0:2].flatten())))
	shower = []
	for i in range(len(segments)-1):
		seg = [segments[i], segments[i+1]]
		energy = 0
		for track in tracklist:
			if(seg[0]>=track[0] and seg[1]<=track[1]): energy+=track[2]
		shower.append([seg[0], seg[1], energy/1000])
	
	#plot
	path = []
	E_visible = []
	for item in shower:
		if(item[1]-item[0]<0.01): continue   #ignore very short segments (<0.01m)
		if(abs(item[0]-dist2max)>100): continue   #ignore anything 100m away from shower_max
		path.append(item[0])
		path.append(item[1])
		E_visible.append(item[2])
		E_visible.append(item[2])
	plt.figure(dpi=150)
	plt.plot(path, E_visible)
	plt.axvline(dist2max, color='r')
	plt.xlabel('Distance from Creation (m)')
	plt.ylabel('Visible Energy (TeV)')
	plt.show()
	
#given two numpy arrays of same length
#return the degree angle between them, using the dot product formula
def dotprod_angle(u, v, axis=0):
	if(axis==0): return np.rad2deg(np.arccos(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)), -1, 1)))
	else: return np.rad2deg(np.arccos(np.sum(u*v,axis=axis)/(np.linalg.norm(u, axis=axis)*np.linalg.norm(v, axis=axis))))

#input: I3 pulse series, such as q['InIcePulses'].apply(q)
#return localness, totalPEs, MSS, MSD, PEmap
def localness_calculator(data, zmap, deep_strings, neighbor_map, MSS_thres=1.5):
	#build PEmap - a 2D numpy array of collected PEs of each DOM -- PEmap[string][DOM]
	PEmap = np.zeros((79, 61))
	for sensor in data:
		if(sensor[0].string in deep_strings): continue   #ignore deep core for now
		PEmap[sensor[0].string, sensor[0].om] = sum([pulse.charge for pulse in sensor[1]])
	PEmap_string = np.sum(PEmap, axis=1)   #1D numpy array of collected PEs of each string
	totalPEs = np.sum(PEmap_string)
	
	#Most Significant Neighborhood -- contains most number of PEs
	MSS_candidates = np.argsort(PEmap_string)[::-1][0:3]   #three brightest strings
	neighborhood_PEs = [PEmap_string[string] + \
						np.sum(PEmap_string[neighbor_map[string][:,0].astype('int')]) for string in MSS_candidates]
	MSS = MSS_candidates[0]                                      #Default MSS is the brightest string
	if(max(neighborhood_PEs) > MSS_thres*neighborhood_PEs[0]):   #unless its neighborhood is not very bright
		MSS = MSS_candidates[np.argmax(neighborhood_PEs)]
		#print 'Warning: MSS not the brightest string.', neighborhood_PEs[0], max(neighborhood_PEs)
		
	#Most Significant DOM -- brightest DOM on the MSS
	MSD = np.argmax(PEmap[MSS])
	#from MSD compute -10+9 layers
	DOM_bounds = [max(MSD-10, 1), min(MSD+9, 60)]   #taken at the MSS
	z_bounds = [zmap[MSS, DOM_bounds[1]], zmap[MSS, DOM_bounds[0]]]
	#localness - PEs in the box
	PEs_box = np.sum(PEmap[MSS, DOM_bounds[0]:DOM_bounds[1]+1])   #start with MSS
	for string in neighbor_map[MSS][:,0].astype('int'):       #then all its neighbors
		DOMs_inbound = np.logical_and(z_bounds[0]<=zmap[string], zmap[string]<=z_bounds[1])
		PEs_box += np.sum(PEmap[string, DOMs_inbound])
		
	return PEs_box/totalPEs, np.max(PEmap)/totalPEs, totalPEs, MSS, MSD, PEmap

#Take pulse series in icetray format, convert to my numpy-based format for easy access
def build_I3dict(data, deep_strings):
	#DOMdict - PEmap plus timing
	#stringdict - DOMdict summed over string by string
	#charging_curve - DOMdict summed over the whole detector
	DOMdict = dict([(string, {}) for string in range(1,79)])
	stringdict = dict([(string, []) for string in range(1,79)])
	charging_curve = []
	for sensor in data:
		if(sensor[0].string in deep_strings): continue   #ignore deepcore, for now
		waveform = [[pulse.time, pulse.charge] for pulse in sensor[1]]
		if(len(waveform)==0): continue
		stringdict[sensor[0].string]+=waveform
		charging_curve+=waveform
		waveform = np.array(waveform)
		waveform[:,1] = np.cumsum(waveform[:,1])   #decided to store the cumsum of the PEseries
		DOMdict[sensor[0].string][sensor[0].om] = waveform
	for string in stringdict:   #this loop for finishing building stringdict
		if(len(stringdict[string])):
			stringdict[string] = np.array(stringdict[string])
			stringdict[string] = stringdict[string][np.argsort(stringdict[string][:,0])]   #sort by time
			stringdict[string][:,1] = np.cumsum(stringdict[string][:,1])
			to_keep = list(np.diff(stringdict[string][:,0])!=0)+[True]
			stringdict[string] = stringdict[string][to_keep]   #get rid of repeated times
	charging_curve = np.array(charging_curve)
	charging_curve = charging_curve[np.argsort(charging_curve[:,0])]   #sort by time
	charging_curve[:,1] = np.cumsum(charging_curve[:,1])
	charging_curve = charging_curve[list(np.diff(charging_curve[:,0])!=0)+[True]]   #get rid of repeated times
	return DOMdict, stringdict, charging_curve
	
#take DOMdict and build fstHit, fstHit_string, z_time, z_charge
#fstHit_string ideally should be determined by the hyperbolic trend in the fstHit of DOMs on the string
#but individual DOMs on a dimly-lit string might not have enough PEs for reliable fstHit on their own
#in that case, determine fstHit_string from the waveform of the whole string
def build_fstHit(DOMdict, stringdict, PEmap, zmap, DOM_thres, numlayers):
	fstHit = np.full((79, 61, 4), np.inf)    #string, DOM, fstHit derivatives
	fstHit_raw = np.full((79, 61), np.inf)    #string, DOM -- ignore DeepCore
	fstHit_string = np.full((79, 4), np.inf)
	z_time = np.full(79, np.inf)
	z_charge = np.full(79, np.inf)
	for string in DOMdict:
		if(not len(DOMdict[string])): continue   #skip empty string
		for sensor in DOMdict[string]:
			fstHit_raw[string][sensor] = DOMdict[string][sensor][0][0]
			if(PEmap[string][sensor]<DOM_thres): continue
			fstHit[string][sensor] = rising_edge_fitter(DOMdict[string][sensor], width_defn=0.5)[0]
		midDOMs = np.argsort(PEmap[string])[-np.sum(PEmap[string]>0.)//3:]   #top third
		fstDOM = midDOMs[np.argmin(fstHit[string,midDOMs,0])]
		z_time[string] = zfitter2(fstDOM, zmap[string], fstHit[string,:,0], 2*numlayers+1)[0]
		z_charge[string] = zfitter2(np.argmax(PEmap[string]), zmap[string], PEmap[string], 2*numlayers+1)[0]#-zcharge_offset
		#fstHit_string[string] = min(np.min(fstHit[string][midDOMs]), first_hit(stringdict[string], fitpower=2)[0])
		fstHit_string[string] = rising_edge_fitter(stringdict[string], width_defn=0.5)[0]
	return fstHit, fstHit_raw, fstHit_string, z_time, z_charge

#scipy.optimize.curve_fit() to fit a hyperbola to the fstHit of brightest DOMs on the string
#to estimate tc, l2, zc of the cascade
#PEmap: totalPEs of each DOM on the string
#fstHit: first hit on each DOM on the string
#zmap: z-coordinate of each DOM on the string
#v: speed of photon in ice, default 0.222m/ns
def hyperbolic_fit(PEmap, fstHit, zmap, v=0.222, part=3, bounds=([5000, 0, -520],[30000, 15000, 520])):
	hyperbola = lambda z, tc, l2, zc: tc + np.sqrt(l2+(z-zc)**2)/v
	midDOMs = np.argsort(PEmap)[-np.sum(PEmap>0.)/part:]   #default part -> use top third DOMs
	midDOMs = midDOMs[np.isfinite(fstHit[midDOMs])]
	while(len(midDOMs)<4 and part>2):
		part-=1
		midDOMs = np.argsort(PEmap)[-np.sum(PEmap>0.)/part:]  #want at least 4 DOMs in fit
		midDOMs = midDOMs[np.isfinite(fstHit[midDOMs])]
	return scipy.optimize.curve_fit(hyperbola, zmap[midDOMs], fstHit[midDOMs], bounds=bounds)[0]   #[tc, l2, zc]
	
#determine xy of shower_max by minimizing error of fitting fstHits_string to d=vt
#need the first element of events to be of MSS
#events: [fstHit, x,y]
def fstHit_xy(events, guess=[], bounds=[]):
    if(len(guess)==0):  #if guess is not provided, make one based on the MSS and the earliest two strings
        E1E2 = events[1:][np.argsort(events[1:][:,0])][:2]   #two earliest strings other than MSS
        weights = 0.5*(1-0.2*(E1E2[:,0]-events[0,0])/np.linalg.norm(E1E2[:,1:], axis=1))
        timepos = np.average(E1E2[:,1:]*np.array([weights]).T, axis=0)
        guess = [np.min(events[:,0])-200]+list(timepos)+[-10**-4, 0.222]
    if(len(bounds)==0):
        bounds = [[guess[0]-300]+list(timepos-50)+[-10**-3, 0.22], 
                  [guess[0]]    +list(timepos+50)+[0., 0.25]]
    
    def err_func(p):
        dt = events[:,0]-p[0]
        dr = np.linalg.norm(events[:,1:]-p[1:3], axis=1)
        v = p[-2]*dr+p[-1]
        return np.sum(np.abs(v*dt-dr))
    min1 = scipy.optimize.differential_evolution(err_func, bounds=np.stack(bounds, axis=1))
    min2 = scipy.optimize.dual_annealing(err_func, bounds=np.stack(bounds, axis=1), maxiter=1000)
    if(err_func(min1.x)<=err_func(min2.x)): return min1
    else: return min2

#generic permutation function
#given a list, return the list of all possible permutations of the given list
def permutation(a):
    base = [[a[0]]]
    for i in range(1,len(a)):
        newbase=[]
        for member in base:
            for j in range(len(member)+1):
                newbase.append(member[0:j]+[a[i]]+member[j:])
        base = newbase
    return base
	
#determine xy of charge center of cascade by minimizing error of fitting PEmap_string to d = mlogPE + b - c*log(d)
#need the first element of events to be of MSS
#events: [PEs, x,y]
def charge_xy(events, guess=[], bounds=[]):
	if(len(guess)==0):
		normfactor = np.sum(np.sqrt(events[:,0][1:]))
		PEpos = np.sum(events[:,1:][1:]*np.reshape(np.sqrt(events[:,0][1:]),(-1,1)), axis=0)/normfactor
		logPE = np.log(events[:,0])
		dist = np.linalg.norm(events[:,1:]-PEpos, axis=1)
		linefit=np.polyfit(logPE, dist, 1)
		guess = np.concatenate((PEpos, linefit, [-linefit[0]]))
	if(len(bounds)==0):
		bounds = [list(PEpos-50)+[10*linefit[0],    0, -0.2*linefit[0]], 
				 list(PEpos+50)+[0.1*linefit[0], 1000, -20*linefit[0]]]
	
	def err_func(p):
		mlogPE = p[-3]*np.log(events[:,0])
		dr = np.linalg.norm(events[:,1:]-p[0:2], axis=1)
		return np.sum(np.abs(mlogPE+p[-2]-p[-1]*np.log(dr) - dr))
	min1 = scipy.optimize.differential_evolution(err_func, bounds=np.stack(bounds, axis=1))
	min2 = scipy.optimize.dual_annealing(err_func, bounds=np.stack(bounds, axis=1), maxiter=1000)
	if(err_func(min1.x)<=err_func(min2.x)): return min1
	else: return min2
	
	
#input events--a 2D array of PEs and string coordinates, and cascade xy location
#return best-fit azmth, scores and azmths
def azmth_fitter(events, shower_max, stringlist, PEmap_string):
	numstrings = len(stringlist)
	dist = np.linalg.norm(events[:,1:]-shower_max, axis=1)
	logPE = np.log(events[:,0])
	slope_dist = np.polyfit(dist, logPE, 1)[0]    #roughly how fast logPE decreases with distance
	
	#first, compute deltadist between each pair of strings
	#then, use it with slope_dist to delta_d adjust the deltalogPE
	deltadist = np.empty((numstrings,numstrings))
	deltalogPE = np.empty((numstrings,numstrings))
	deltalogPE_adj = np.empty((numstrings,numstrings))
	for i in range(numstrings):          #look at each string one by one
		string = stringlist[i]
		deltadist[i] = dist[i]-dist
		deltalogPE[i] = np.log(PEmap_string[string])-np.log(PEmap_string[stringlist])
		deltalogPE_adj[i] = deltalogPE[i] - slope_dist*deltadist[i]   #adjust PE difference due to distance difference
		
	#compute deltaphi at each azmth_guess
	azmths = np.deg2rad(np.arange(0,360))   #work in increment of 1 degree
	deltaphi = np.empty((len(azmths),numstrings,numstrings))
	for i in range(len(azmths)):
		phi = dotprod_angle(events[:,1:]-shower_max, 
							numstrings*[[np.cos(azmths[i]), np.sin(azmths[i])]], axis=1)   #pick azmth, get a phi to each string
		deltaphi[i,:,:] = np.array([phi[j]-phi for j in range(numstrings)])
		
	#compare deltalogPE_adj with deltaphi and generate scores
	slope_phi = -np.max(deltalogPE_adj)/np.max(deltaphi)   #estimate of deltaphi effect on deltalogPE_adj
	scores = np.sum(np.abs(deltaphi*slope_phi - np.array([deltalogPE_adj]*len(azmths))), axis=(1,2))
	
	#extend scores for the purpose of having enough space for fitting
	fit_len = len(azmths)/4   #total length used in fit is actually twice of fit_len
	min_arg = np.argmin(scores)+len(scores)   #the +len(scores) shift anticipates the line below
	scores = list(scores)*3
	azmths = list(azmths-azmths[-1])+list(azmths)+list(azmths+azmths[-1])
	fits = np.polyfit(azmths[min_arg-fit_len:min_arg+fit_len], scores[min_arg-fit_len:min_arg+fit_len], 2)
	azmth_bestfit = -fits[1]/(2*fits[0])
	
	return azmth_bestfit, scores, azmths
	
#given a direction axis=(azmth, znth) and rotation angle a,
#return the rotation matrix
#https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
def rotation_matrix(axis, a):
	x=np.cos(axis[0])*np.sin(axis[1])
	y=np.sin(axis[0])*np.sin(axis[1])
	z=np.cos(axis[1])
	A = np.tile([x,y,z],(3,1))
	B = np.array([[np.cos(a), -z*np.sin(a), y*np.sin(a)],
				[z*np.sin(a), np.cos(a), -x*np.sin(a)],
				[-y*np.sin(a), x*np.sin(a), np.cos(a)]])
	return A*A.T*(1-np.cos(a))+B
	
	
#given the fstHits and DOMcoords, return "trackness"
#trackness--ratio of max over median of the track-equation-fit plot
def trackness_compute(fstHits, DOMcoord, c=0.3, error=0.05, na=90, nz=45):
	finiteDOMs = np.argwhere(np.isfinite(fstHits))#*(PEmap>PEthres))
	delta_s = []
	for num,DOM1 in enumerate(finiteDOMs):
		for DOM2 in finiteDOMs[num+1:]:
			delta_t = fstHits[DOM1[0],DOM1[1]]-fstHits[DOM2[0],DOM2[1]]
			delta_r = DOMcoord[DOM1[0],DOM1[1]]-DOMcoord[DOM2[0],DOM2[1]]
			sign = 1 if delta_t>0 else -1
			delta_s.append([abs(delta_t)]+list(sign*delta_r))    #keep the positive delta_t
	delta_s = np.array(delta_s)
	
	dist_c_min = (c-error)*delta_s[:,0]
	dist_c_max = (c+error)*delta_s[:,0]
	scan = np.zeros((na,nz))
	azmth = np.linspace(0,2*np.pi,na)
	znth = np.pi-np.linspace(0,np.pi,nz)
	scan_znth,scan_azmth = np.meshgrid(znth,azmth)
	v = np.array([np.cos(scan_azmth)*np.sin(scan_znth),
				np.sin(scan_azmth)*np.sin(scan_znth),
				np.cos(scan_znth)])
	for a in range(na):
		for z in range(nz):
			dist = np.sum(delta_s[:,1:]*v[:,a,z],axis=1)
			scan[a,z] = np.sum((dist<=dist_c_max)*(dist>=dist_c_min))   #this is very crude
	scan_reduced = []
	for i,rad in enumerate(np.sin(np.pi-znth)):
		scan_reduced+=list(scan[np.linspace(0, na-1, int(round(na*rad))).astype('int'), i])
	return np.max(scan)/np.median(scan_reduced)
	
#compute trackness given a set of points uniformly distributed on the sphere
#points: (azmth, znth)
#Rmax: meter, search around charge_center
#nr  : 2*Rmax = stepsize*nr   must be odd to include the center point
def trackness_compute2(fstHits, PEmap, DOMcoord, points, c=0.299792458, Rmax=50, nr=5, rho_max=60):
	tanc = 0.8692867378162267   #np.tan(np.deg2rad(41))
	score_eval = lambda error: np.sin(0.5*np.pi*np.exp(-2*error))   #the 2 here is not really optimized
	marks = np.linspace(-Rmax,Rmax,nr)
	scan_prime = np.transpose([np.tile(marks, len(marks)), np.repeat(marks, len(marks))])
	scan_prime = scan_prime[np.linalg.norm(scan_prime, axis=1)<Rmax]   #for moving track's pos on the perpendicular plane
	
	finiteDOMs = np.argwhere(np.isfinite(fstHits)*(PEmap>0))   #DOM ordering here derives from finiteDOMs's
	DOMfstHit = fstHits[finiteDOMs[:,0], finiteDOMs[:,1]]
	DOMpos = DOMcoord[finiteDOMs[:,0], finiteDOMs[:,1]]
	DOMPEs = PEmap[finiteDOMs[:,0], finiteDOMs[:,1]]
	brightestDOM = np.argmax(DOMPEs)
	
	scores = np.empty((len(points)))
	for i, p in enumerate(points):
		axes_prime = np.array(new_axes(p[0], p[1]))      #the last axis is track's direction
		DOMpos_prime = np.dot(DOMpos, axes_prime.T)[:,0:2]  #DOM projection onto the perpendicular plane
		search_area = DOMpos_prime[brightestDOM]+scan_prime    #search_area around brightest DOM
		
		#track equation: c*delta_t = distdotv + delta_rho*tanc
		#search for delta_rho that minimize error: sum(|c*delta_t-distdotv-delta_rho_prime*tanc|)
		max_score = 0
		for loc in search_area:
			rho_prime = np.linalg.norm(DOMpos_prime-loc, axis=1)
			DOMs2use = np.where(rho_prime<rho_max)[0]
			pairs2use = pair_generator2(DOMs2use)
			pairs2use0, pairs2use1 = pairs2use[:,0], pairs2use[:,1]
			delta_rho_prime = rho_prime[pairs2use0]-rho_prime[pairs2use1]
			cdelta_t = c*(DOMfstHit[pairs2use0]-DOMfstHit[pairs2use1])
			delta_r = DOMpos[pairs2use0]-DOMpos[pairs2use1]
			delta_r_mag = np.linalg.norm(delta_r, axis=1)
			distdotv = np.dot(delta_r, axes_prime[-1])
			score = np.sum(score_eval(np.abs(cdelta_t-distdotv-delta_rho_prime*tanc)/delta_r_mag))
			if(score>max_score): max_score=score
		scores[i]=max_score
		
	scores = scores-np.min(scores)   #is it right to do this?
	return np.max(scores)/np.median(scores)

#given a direction in standard spherical (azmth, znth)
#return a new set of axes (orthonormal unit vectors) with the last one being the input direction
def new_axes(azmth, znth):
    #R = rotation_matrix((azmth, znth), np.pi/2)
    cosa, sina = np.cos(azmth), np.sin(azmth)
    cosz, sinz = np.cos(znth), np.sin(znth)
    cosz2, sinz2 = -sinz, cosz   #np.cos(znth+np.pi/2), np.sin(znth+np.pi/2)
    
    v = np.array([cosa*sinz, sina*sinz, cosz])
    a = np.array([cosa*sinz2, sina*sinz2, cosz2])   #a is v with znth shifted by +90
    b = np.cross(v,a) #np.matmul(R, a)
    return (a,b,v)

#given the number of objects, return all possible pairings
#note: total number of pairs given by (n choose 2): scipy.special.factorial(n)/(2*scipy.special.factorial(n-2))
def pair_generator(num):
	total = num*(num-1)//2
	pairs = np.empty((total,2), dtype='int')
	index = 0
	for i in range(num):
		for j in range(i+1,num):
			pairs[index] = (i,j)
			index+=1
	return pairs
#given a list of numbers, return a list of all possible pairings of the numbers
def pair_generator2(nums):
	num = len(nums)
	total = num*(num-1)//2
	pairs = np.empty((total,2), dtype='int')
	index = 0
	for i in range(num):
		for j in range(i+1,num):
			pairs[index] = (nums[i],nums[j])
			index+=1
	return pairs
	
	
#given the number of zenith bins, return a list of (azmth, znth) uniformly distributed on the sphere
def sphere_pixel_gen(nz):
    points=[]
    for z in np.linspace(0, np.pi, nz):
        for a in np.linspace(0, 2*np.pi, 1+np.int(2*nz*np.sin(z))):
             points.append([a,z])
    return np.array(points)
	

#build a table of angles corresponding to f=c*delta_t/delta_r
#i.e. for inverting f=cos(theta)+sin(theta)*cos(alpha)*tan(41)
def build_track_table(f_res=0.001, tanc=np.tan(np.radians(41))):
	func = lambda theta, cosa, f: np.cos(theta)+np.sin(theta)*cosa*tanc-f
	track_table = []
	f_value = np.arange(0,1.326, f_res)
	for val in f_value:
		if(val<1.0):
			theta_min=scipy.optimize.fsolve(func, 0.2, args=(-1,val))[0]
			theta_max=scipy.optimize.fsolve(func, 2.0, args=(+1,val))[0]
		if(val>=1.0):
			theta_min=scipy.optimize.fsolve(func, 0.2, args=(+1,val))[0]
			theta_max=scipy.optimize.fsolve(func, 2.0, args=(+1,val))[0]
		track_table.append([val, theta_min, theta_max])
	return track_table
#given f (inverse trigger speed, c*delta_t/delta_r) and g (rho bias, delta_rho_max/delta_r) of a DOMpair
#return the lower and upper bounds to the track-DOMpair cos(angle)
#A is cosa*tanc, but the default value of cosa is 1
#f_bounds from the track equation at fixed f
#g_bounds from imposing delta_rho_max condition
#h_lower from imposing rho_max condition
#NOTE: valid for nonnegative f only
def track_theta_bounds(f, g=1, tanc=np.tan(np.radians(41))):
	A = tanc
	sec2c = 1+A**2
	discriminant = sec2c-f**2
	discriminant[discriminant<0] = np.nan   #this line solely to prevent the warning message
	radical = A*np.sqrt(discriminant)
	f_lower = (f-radical)/sec2c
	f_upper = (f+radical)/sec2c
	g_lower = f-g*tanc
	g_upper = f+g*tanc
	h_lower = np.sqrt(np.clip(1-4*g*g, 0, None))   #this is wrong, but to do it right involves dealing with multiple bands
	cos_lower = np.fmax(f_lower, np.fmax(g_lower, h_lower))   #fmax ignores nan
	cos_upper = np.fmin(f_upper, g_upper)                     #fmin ignores nan
	clean = (cos_lower>=cos_upper)+np.isnan(discriminant)
	cos_lower[clean] = np.nan
	cos_upper[clean] = np.nan
	
	theta_lower = np.arccos(cos_upper)
	theta_upper = np.arccos(cos_lower)
	return theta_lower, theta_upper
	

#Given an I3 event, pick out DOMs that line up on a track pattern.
#Do this and determine a track length, preVoid, and postVoid at every sky pixel.
#the ideal track:
#-Many DOMs triggered in succession in a straight line, at 0.3m/ns
#-All possible pairings of the DOMs are valid, obeying the track equation, all (numDOMs choose 2) of them
#-Long track length, over twice string spacings
#tanc=np.tan(np.radians(41))
def track_detector(fstHits, PEmap, DOMcoord, scatmap, MSS, MSD, stringlist, boundary_strings,
					NSIDE=16, max_deltzp=160., rho_factor=1., PEthres=0.3, c=0.299792458, c_error=0.03, thresVoid=100):
	dustlayer = [-150, 0]   #z coodinates in meter, values chosen so layer as thick as min search windows
	tcore = max(fstHits[0,0], fstHits[MSS, MSD])   #time dividing the event into preMSD and postMSD
	MSDpos = DOMcoord[MSS, MSD]   #best estimate of cascade's location
	for _ in (True, False):
		finiteDOMs = np.argwhere(np.isfinite(fstHits)
								*(np.abs(fstHits-tcore)<6000)   #max DOMpair length: 1557m
								*(PEmap>PEthres))
		DOMfstHit = fstHits[tuple(finiteDOMs.T)]
		timeorder = np.argsort(DOMfstHit)
		finiteDOMs = finiteDOMs[timeorder]   #reorder finiteDOMs, earliest to latest, ensuring cdelta_t is always positive
		DOMfstHit = DOMfstHit[timeorder]
		DOMpos = DOMcoord[tuple(finiteDOMs.T)]
		coreDOMs = set(np.flatnonzero((np.linalg.norm(DOMpos-MSDpos, axis=1)<40)))   #allow for two DOMs above and two DOMs below
		if(coreDOMs): break
		else: fstHits[MSS, MSD] = fstHits[0,0]   #if no coreDOM, force the MSD to be one
	DOMPEs = PEmap[tuple(finiteDOMs.T)]
	DOMoutside = (np.linalg.norm(DOMpos-MSDpos, axis=1)>160) + \
				 [sensor[0] in boundary_strings or sensor[1]<4 for sensor in finiteDOMs]   #more than 160m from MSD, or at boundary
	DOMscat = np.clip(scatmap[tuple(finiteDOMs.T)], 15, None)
	sumLogPEs = np.sum(np.log10(1+DOMPEs[np.linalg.norm(DOMpos-MSDpos, axis=1)<500]))
	
	#pair up DOMs
	numDOMs = len(finiteDOMs)
	pairs = pair_generator(numDOMs)   #all DOMpairs, DOM indices derived from finiteDOMs
	cdelta_t = c*(DOMfstHit[pairs[:,1]]-DOMfstHit[pairs[:,0]])
	delta_r = DOMpos[pairs[:,1]]-DOMpos[pairs[:,0]]
	delta_r_mag = np.linalg.norm(delta_r, axis=1)
	pairdir = (delta_r.T/delta_r_mag).T   #cartesian
	delta_rho_max = rho_factor*np.min(DOMscat[pairs], axis=1)   #for cutting off DOMpairs with too large delta_rho or too large rhos
	
	#Tabulate valid DOMpairs at every pixel in the sky
	F = cdelta_t/delta_r_mag   #the central piece of this method
	theta_min, theta_max = track_theta_bounds(F, g=delta_rho_max/delta_r_mag, tanc=0.8692867378162267)
	NPIX = hp.nside2npix(NSIDE)   #number of sky pixels
	skypixels_pairs = np.full((NPIX, len(pairs)), False)   #masks giving compatible DOMpairs at each pixel
	for i,vec in enumerate(pairdir):
		if(np.isnan(theta_max[i])): continue   #(can also be theta_min)
		inner_bound = hp.query_disc(nside=NSIDE, vec=vec, radius=theta_min[i])
		outer_bound = hp.query_disc(nside=NSIDE, vec=vec, radius=theta_max[i])
		skypixels_pairs[outer_bound, i] = True
		skypixels_pairs[inner_bound, i] = False
		
	#Build the best track candidate at each pixel, by assembling the largest set of (mostly) connected DOMpairs
	#identify the most connected DOMpair -- the two DOMs forming the most number of valid pairs with other DOMs in the pixel
	hotpixels = np.full((NPIX, 8), np.nan)
	vetoDOMs_dict = {}
	i0 = 0   #NSIDE*(6*NSIDE-2):start of the not upgoing pixels, NSIDE*(6*NSIDE+2):start of downgoing pixels
	for i, mask in enumerate(skypixels_pairs[i0:], i0):   
		pairs_at_pixel = pairs[mask]     #at each pixel, a list of DOMpairs, each with f value compatible with the pixel
		if(len(pairs_at_pixel)==0): continue   #done if an empty pixel
		DOMs0,DOMs1 = pairs_at_pixel.T
		maxnum=0   #max size(number of pairs) of track candidate
		DOMconxns = {DOM : {DOM}|set(DOMs0[DOMs1==DOM])|set(DOMs1[DOMs0==DOM]) \
											for DOM in np.unique(pairs_at_pixel)}   #DOMs that each DOM connects to in the pixel
		DOMsABs = [DOMconxns[ab[0]]&DOMconxns[ab[1]] for ab in pairs_at_pixel]   #DOMs that each DOMpair connects to
		DOMsABs_len = [len(DOMsAB) for DOMsAB in DOMsABs]
		for j in np.argsort(DOMsABs_len)[::-1]:   #find the main DOMpair of the pixel
			if(DOMsABs_len[j]*(DOMsABs_len[j]-1)/2 <= maxnum): break   #done as soon as no sufficiently large DOMsAB left
			if(not DOMsABs[j]&coreDOMs): continue   #DOMsPixel must contain at least one of the coreDOMs
			DOMsAB = list(DOMsABs[j])
			DOMsAB_mask = np.full(numDOMs, False)
			DOMsAB_mask[DOMsAB] = True
			numpairsAB = np.sum(DOMsAB_mask[DOMs0]*DOMsAB_mask[DOMs1])   #count number of connections among the chosen DOMs
			if(numpairsAB>maxnum):
				maxnum = numpairsAB
				DOMsPixel = np.array(DOMsAB)
		if(not maxnum): continue
		
		#examine trackchain, look for weaknesses in gaps -- if too large, cut out
		axes_prime = np.array(new_axes(*hp.pix2ang(NSIDE, i)[::-1]))
		zpPixel = np.dot(DOMpos[DOMsPixel], axes_prime[2])   #zprime of every DOM in the DOMsPixel
		pospMSD = np.dot(MSDpos, axes_prime.T)
		order = np.argsort(zpPixel)
		zpPixel = zpPixel[order]       #ordered by zprime
		DOMsPixel = DOMsPixel[order]   #ordered by zprime
		sp = np.argmin(np.abs(zpPixel-pospMSD[2]))   #splitpoint
		gaps1, gaps2 = np.diff(zpPixel[:sp+1]), np.diff(zpPixel[sp:])
		gaps1_rules = max_deltzp*np.clip(np.arange(1,len(gaps1)+1), None, 4)
		gaps2_rules = max_deltzp*np.clip(np.arange(1,len(gaps2)+1)[::-1], None, 4)
		wps1, wps2 = np.flatnonzero(gaps1>gaps1_rules), np.flatnonzero(gaps2>gaps2_rules)   #weakpoints
		wps1 = [p for p in wps1 if nondust_frac(DOMpos[DOMsPixel[p]][2], DOMpos[DOMsPixel[p+1]][2], dustlayer)* \
									gaps1[p]>gaps1_rules[p]]   #pardon weakpoints in dustlayer
		wps2 = [p for p in wps2 if nondust_frac(DOMpos[DOMsPixel[sp+p]][2], DOMpos[DOMsPixel[sp+p+1]][2], dustlayer)* \
									gaps2[p]>gaps2_rules[p]]   #pardon weakpoints in dustlayer
		cutpoint1 = max(wps1)+1 if(len(wps1)) else None
		cutpoint2 = min(wps2)+1+len(gaps1) if(len(wps2)) else None
		DOMsPixel = DOMsPixel[cutpoint1:cutpoint2]   #cut
		if(len(DOMsPixel)<4): continue
		zpPixel = zpPixel[cutpoint1:cutpoint2]       #cut
		
		#note down noteworthy pixels - if the DOMsubset fits with a speed close enough to c
		tPixel = DOMfstHit[DOMsPixel]
		polyfit = np.polyfit(tPixel, zpPixel, 1)
		if(abs(polyfit[0]-c) <= c_error):
			
			#determine burdenDOMs based on the xyprime spread of DOMsPixel
			DOMposp = np.dot(DOMpos, axes_prime.T)
			xypPixel = DOMposp[:,:2][DOMsPixel]
			xypbounds = [np.min(xypPixel, axis=0), np.max(xypPixel, axis=0)]
			margins = np.maximum(70-(xypbounds[1]-xypbounds[0])/2, [10,10])   #make sure search window is at least 140m wide
			xypbounds[0]-=margins
			xypbounds[1]+=margins
			if(not np.all((pospMSD[:2]>xypbounds[0])*(pospMSD[:2]<xypbounds[1]))): continue   #check MSD location close to track
			DOMcoord_prime = np.dot(DOMcoord, axes_prime.T)
			with np.errstate(invalid='ignore'):
				burdenZps = DOMcoord_prime[:,:,2][(DOMcoord_prime[:,:,0]>xypbounds[0][0])*(DOMcoord_prime[:,:,0]<xypbounds[1][0]) \
												 *(DOMcoord_prime[:,:,1]>xypbounds[0][1])*(DOMcoord_prime[:,:,1]<xypbounds[1][1]) \
												 *((DOMcoord[:,:,2]<dustlayer[0])+(DOMcoord[:,:,2]>dustlayer[1]))]   #no burdenDOM in dust layer
			preEdge, postEdge = (np.min(burdenZps), np.max(burdenZps)) if(len(burdenZps)) else (np.nan, np.nan)
			preVoid, postVoid = zpPixel[0]-preEdge, postEdge-zpPixel[-1]
			
			#count DOMs triggering before and far away from the MSD, but still somewhat close to the polyfit line
			zpt_bound = 2*np.max(np.abs(zpPixel-polyfit[0]*tPixel - polyfit[1]))
			preDOMs_out = np.flatnonzero((DOMfstHit<tcore)*(DOMposp[:,2]<pospMSD[2]) \
										 *(DOMposp[:,0]>xypbounds[0][0])*(DOMposp[:,0]<xypbounds[1][0]) \
										 *(DOMposp[:,1]>xypbounds[0][1])*(DOMposp[:,1]<xypbounds[1][1]) \
										 *DOMoutside)
			vetoDOMs = preDOMs_out[np.abs(DOMposp[:,2][preDOMs_out]-polyfit[0]*DOMfstHit[preDOMs_out]-polyfit[1]) < zpt_bound]   #still somewhat close
			
			#record the hotpixel
			hotpixels[i] = i, zpPixel[-1]-zpPixel[0], preVoid, postVoid, len(vetoDOMs), len(DOMsPixel), pospMSD[2]-preEdge, polyfit[0]/c-1
			vetoDOMs_dict[i] = vetoDOMs
	
	#find the best pixel: minimal total void length
	#4. through-going track : prefull and postfull
	#3. stopping track: prefull and smallest postVoid
	#2. starting track: postfull and smallest preVoid
	#1. cascade: smallest preVoid
	hotpixels = hotpixels[np.isfinite(hotpixels[:,0])]   #cut out all the unused pixels
	if(not len(hotpixels)): return np.array([-1, 0., 2000, 2000, 0, 0, 2000, -1]), 0, sumLogPEs, np.array([]), np.array([])
	#find the best pixel based on: void score, fit speed, track length, DOMsPixel,
	#the best pixel passes all the tests, and is most centered among other qualifying pixels
	#score: 4:through-going track, 3:stopping track, 2:starting track, 1:cascade, 0:no hotpixel
	hotpixels = hotpixels[np.isfinite(hotpixels[:,0])]   #cut out all the unused pixels
	scores = 1 + 1*(hotpixels[:,3]<thresVoid) + 2*(hotpixels[:,2]<thresVoid)
	test_DOMsPixel = hotpixels[:,5]>=np.quantile(hotpixels[:,5], 0.5)   #top number of DOMs in the pixel
	test_tracklength = hotpixels[:,1]>=np.quantile(hotpixels[:,1], 0.5)   #top lateral span of DOMs in the pixel
	test_speed = np.abs(hotpixels[:,7])<=np.quantile(np.abs(hotpixels[:,7]), 0.5)   #closest fit speed to c
	test_score = scores <= np.quantile(scores, 0.5)   #top score
	test_overall = (1*test_DOMsPixel+1*test_tracklength+1*test_speed+1*test_score)>2
	if(not np.sum(test_overall)): return np.array([-1, 0., 2000, 2000, 0, 0, 2000, -1]), 0, sumLogPEs, np.array([]), np.array([])
	final_dirs = np.array(hp.pix2vec(NSIDE, hotpixels[test_overall][:,0].astype('int'))).T
	angdists = [np.arccos(np.clip(np.sum(np.full(final_dirs.shape, vec)*final_dirs, axis=1),-1,1)) for vec in final_dirs]
	angdists_argmin = np.argmin(np.median(angdists, axis=1))
	best_pixel = hotpixels[test_overall][angdists_argmin]   #choose the most centered of the qualifying pixels
	score = scores[test_overall][angdists_argmin]
	
	return best_pixel, score, sumLogPEs, finiteDOMs[vetoDOMs_dict[best_pixel[0]]], hotpixels[test_overall][:,0].astype('int')
	
	
#given a direction(cartesian) and a pos, definition of detector's boundary
#return an estimate of the tracklength that cuts through the detector
def tracklength_estimator(vec, pos, xybounds, zbounds):
	if(np.any(np.isnan(vec+pos))): return np.nan, [[np.nan]*3]*2
	r_z = np.sort((zbounds-pos[2])/vec[2]) if(vec[2]!=0.) else np.array([-np.inf, np.inf])   #full distances from pos, along vec, to z boundaries of detector
	
	vecxy = vec[0:2]/np.linalg.norm(vec[0:2])
	U = np.array([[vecxy[0], vecxy[1]],
				  [-vecxy[1], vecxy[0]]])   #2x2 matrix for rotating (in the xy plane) so that track aligns with x-axis
	xyboundsU = np.matmul(U,xybounds.T).T - np.matmul(U,pos[0:2])  #rotate and translate
	interxns = np.flatnonzero(np.sign(xyboundsU[:,1]) != np.sign(np.roll(xyboundsU[:,1],-1)))
	a = xyboundsU[interxns].T                    #"left" vertex to the intersection
	b = xyboundsU[(interxns+1)%len(xyboundsU)].T #"right" vertex to the intersection
	rho_xy = (b[1]*a[0]-b[0]*a[1])/(b[1]-a[1])   #horizontal distances from pos, along vec, to xy boundaries of detector
	r_xy = np.sort(rho_xy/np.linalg.norm(vec[0:2]))  #full distances from pos, along vec, to xy boundaries of detector
	if(len(r_xy)!=2): r_xy = np.array([np.nan, np.nan])

	r = np.minimum(abs(r_xy), abs(r_z))*[-1,1]
	r_intercepts = pos + vec*np.reshape(r,(-1,1))   #the two points (standard I3 coordinate) where the track crosses boundaries of the detector
	
	return np.linalg.norm(r_intercepts[1]-r_intercepts[0]), r_intercepts   #an estimate of the tracklength inside the detector
#---------depreciated------------------------------------------------------
#based on identifying two closest boundary DOMs to the track
def tracklength_estimator_old(vec, pos, boundaries, DOMcoord, max_rho=70):
	xyprime = [np.dot(pos, vec[0]), np.dot(pos, vec[1])]
	candidates_rho = []
	candidates_zp = []
	for side in boundaries:
		boundary_DOMs = boundaries[side]
		boundary_DOMs_pos = DOMcoord[boundary_DOMs[:,0], boundary_DOMs[:,1]]
		xp = np.dot(boundary_DOMs_pos, vec[0]) - xyprime[0]
		yp = np.dot(boundary_DOMs_pos, vec[1]) - xyprime[1]
		zp = np.dot(boundary_DOMs_pos, vec[2])
		boundary_DOMs_rho = np.sqrt(xp*xp + yp*yp)
		DOM_indx = np.argmin(boundary_DOMs_rho)   #the index of the candidate DOM from that side
		candidates_rho.append(boundary_DOMs_rho[DOM_indx])
		candidates_zp.append(zp[DOM_indx])
	candidates_zp = np.array(candidates_zp)[np.array(candidates_rho)<max_rho]
	return np.max(candidates_zp)-np.min(candidates_zp)
	

#input an icemodel.dat and DOMcoord (ex. combo\src\ice-models\resources\models\spice_3.2.2\icemodel.dat)
#return scattering length at where the DOMs are, [z, scat_len], and [z, absp_len]
def icemodel(filepath, DOMcoord, z_origin=1948.07):
	f = open(filepath, 'r')
	icemodel=np.array([[float(item) for item in line.split(' ')[0:3]] for line in f])
	f.close()
	icemodel[:,0] = z_origin-icemodel[:,0]
	icemodel[:,1] = 1/icemodel[:,1]   #scattering length
	icemodel[:,2] = 1/icemodel[:,2]   #absorption length
	scatmap = np.full((79, 61), np.nan)
	z = icemodel[:,0][::-1]
	s = icemodel[:,1][::-1]
	for string in range(1,79):
		for DOM in range(1,61):
			scatmap[string,DOM] = linear_read(DOMcoord[string,DOM][2], z, s)
	
	return scatmap, icemodel[:,[0,1]], icemodel[:,[0,2]]
	
	
#given the year and date(four digits, ex. 0101-->Jan01), return a dict of I3file and GCD directories
def I3filedir_gen(year, date):
	#---Parent directory--------------------------------------------
	data_dir = '/data/exp/IceCube/'+year+'/filtered/level2'+'/'+date
	if(not os.path.isdir(data_dir)): return {}
	#---------------------------------------------------------------
	
	#Get list of runs in data_dir
	runlist = []
	for item in os.listdir(data_dir):
		if(item[0:3]=='Run' and len(item)==11 and not '_' in item): runlist.append(item)
	runlist = sorted(runlist)
	
	#loop through each run in runlist:
	filedir_dict = {'sorted_runs':runlist}
	for run in runlist:
		filelist = []
		for item in os.listdir(data_dir+'/'+run):
			if(len(item)==64 and '.i3.zst' in item and not '_IT' in item): filelist.append(item)
			if('_GCD.i3.zst' in item): newGCD = item
		filedir_dict[run] = sorted(filelist)
		filedir_dict[run+'_dir'] = data_dir+'/'+run
		filedir_dict[run+'_GCD'] = filedir_dict[run+'_dir']+'/'+newGCD   #new run, new GCD
	return filedir_dict


#inputs: MSD, PEmap[string], fstHit[string], DOMcoord[MSS,:,2]
#pick 7 DOMs to fit based on PEmap, centered around the MSD
def cascadeZestimator(MSD, DOMzs, chargedata, timedata):
	with np.errstate(divide='ignore'): chargedata = np.log(chargedata)   #fit log(PE) actually
	DOMs2fit = [MSD]
	numDOMsleft = 6
	level=1   #try to build list of DOMs to fit, using DOMs closest as possible to MSD
	while(numDOMsleft>0 and level<10):   #the 10 is kind of arbitrary
		if(MSD-level>0 and np.isfinite(chargedata[MSD-level])):   #if haven't passed top of string
			DOMs2fit.append(MSD-level)
			numDOMsleft-=1
		if(MSD+level<61 and np.isfinite(chargedata[MSD+level])):
			DOMs2fit.append(MSD+level)
			numDOMsleft-=1
		level+=1
	DOMzQ = DOMzs[DOMs2fit[np.argmax(chargedata[DOMs2fit])]]
	DOMzT = DOMzs[DOMs2fit[np.argmin(timedata[DOMs2fit])]]
	if(len(DOMs2fit)<3): return np.nan, DOMzQ-DOMzT
	chargefit = np.polyfit(DOMzs[DOMs2fit], chargedata[DOMs2fit], 2)
	timefit = np.polyfit(DOMzs[DOMs2fit], timedata[DOMs2fit], 2)
	zcharge = -chargefit[1]/(2*chargefit[0])
	ztime = -timefit[1]/(2*timefit[0])
	return(zcharge-ztime, DOMzQ-DOMzT)


#given z coordinates of two points and of the dust layer,
#return fraction of the delta z that is outside the dust layer, clipped at minimum 0.3
def nondust_frac(a, b, dustlayer):
    if(np.isclose(a,b)): return max(1.0-float(dustlayer[0]<a<dustlayer[1]), 0.3)
    minz, maxz = sorted([a, b])
    if(maxz<dustlayer[0] or minz>dustlayer[1]): return 1.0
    deltz_dust = min(maxz, dustlayer[1]) - max(minz, dustlayer[0])
    return max(1.0-deltz_dust/(maxz-minz), 0.3)


#given the Most Significant String (MSS), find and return a chain of DOMs above and before the cascade
#the chain of DOMs is strong evidence of the presence of an entering muon
def MSS_veto(fstHit_MSS, MSD, tcore, zcoord_MSS):
	vetoDOMs = np.sort(np.flatnonzero(((fstHit_MSS>tcore-1000)
										*(fstHit_MSS<tcore))[:MSD]))   #Early triggering DOMs above the MSD
	if(not len(vetoDOMs)): return []
	vetoDOMs = vetoDOMs[fstHit_MSS[vetoDOMs]<(np.median(fstHit_MSS[vetoDOMs]) + tcore)/2.]   #further thin out vetoDOMs
	if(not len(vetoDOMs)): return []
	tails = {DOM: [] for DOM in vetoDOMs}   #each vetoDOM and all DOMs above that connects* to it
	maxtail = {DOM: [None, 0] for DOM in vetoDOMs}  #each vetoDOM and its longest tailDOM
	for a,b in pair_generator2(vetoDOMs):
		if(abs(b-a)>5): continue   #don't look at pairs that are too long--more than 5 DOM spacings
		F = (fstHit_MSS[a]-fstHit_MSS[b])/(zcoord_MSS[b]-zcoord_MSS[a])
		if(F<0 or F>5):continue   #*connects, 1/5 = 0.2m/ns
		tails[b].append(a)
	for DOM in vetoDOMs[1:]:   #this loop for filling maxtail
		for tail in tails[DOM]:
			if(maxtail[tail][1]+1>maxtail[DOM][1]):
				maxtail[DOM] = [tail, maxtail[tail][1]+1]
	
	#find the longest chain ending in one of the coreDOMs
	coreDOMs = [DOM for DOM in [MSD-1, MSD-2, MSD-3] if DOM in vetoDOMs]
	if(not coreDOMs): return []
	theDOM = coreDOMs[np.argmax([maxtail[key][1] for key in coreDOMs])]
	chain = [theDOM]
	while(maxtail[chain[-1]][1]!=0):
		chain.append(maxtail[chain[-1]][0])
	
	return chain


#estimate cascade's xy location from the collected charges
def xyEstimate(stringlist, PEmap_stringlist, stringcoord):
	finiteStrings = np.flatnonzero(PEmap_stringlist>1)
	coords, weights = stringcoord[finiteStrings], np.log(PEmap_stringlist[finiteStrings])
	return np.sum(coords*np.tile(weights, (2,1)).T, axis=0)/np.sum(weights)
	
	
#build E(z') of a track event (which can be made up of multiple muons) from I3MCTree and MMCTrack
#points on z' chosen with default spacing of 1m, starts at event's entrance point into detector, ends at exit point
#energy read at each chosen point, by summing up contribution from every muon at that point
#return E(z') represented by an array of energies and corresponding I3 coordinates
def track_energies(MCTree, MMCTrackList, spacing=1):
	surface = MuonGun.Cylinder(1100,600)   #define detector's boundary -- a cylinder of height 1100m, radius 600m
	primary = MCTree.get_head()
	intersections = surface.intersection(primary.pos, primary.dir)
	event_dir = np.array(1*primary.dir)
	dists = np.arange(intersections.first, intersections.second, spacing)   #points on the track to read the energy at
	event_coords = np.tile(primary.pos,(len(dists),1))+ \
					np.tile(dists, (3,1)).T*np.tile(event_dir, (len(dists),1))   #an array of I3 coordinates of points on the track
	event_energies = np.zeros(len(dists))
	for track in MuonGun.Track.harvest(MCTree, MMCTrackList):
		track_dir = np.array(1*track.dir)
		track_dists = np.dot(event_coords, track_dir) - np.dot(track.pos, track_dir)
		track_energies = [track.get_energy(x) for x in track_dists]
		event_energies+=track_energies
		
	return event_coords, event_energies


#A better method than in track_detector (DOMpair analysis)
#Detects a track in an IceCube event by scanning every pixel of the sky - at each pixel a cylindar, a few hundred meters radius and height,
#moving at c through the detector, direction of the pixel, and centered around the brightest DOM (the MSD). Any DOM triggering while inside the
#cylindar is collected. Chain those DOMs together and see if they spread over the detector from edge to edge, and fit to speed c.
def track_sweeper(fstHits, tcore, PEmap, DOMcoord, MSS, MSD, boundary_strings, DOMdict, table_smoothing, clearDOMs,
                  NSIDE=16, cylinder=(160., 240.), PEthres=0.5, thresVoid=150, c=0.299792458, cmin=0.15):
	dustlayer = [-150, 0]   #z coodinates in meter, values chosen so layer as thick as search windows
	max_deltzp = 160.   #for trimming DOMchain to calculate tracklength, based on max string distance, 160m is IceCube's max string spacing
	outdefn = 160.       #m, define boundary of cascade, whether the deposited PE is considered part of the cascade or the precursor track
	cosz_thres = -0.139   #82degrees, threshold of cosine zenith for considering the dust layer for down going events
	
	MSDpos = DOMcoord[MSS, MSD]   #best estimate of cascade's location
	finiteDOMs = np.argwhere(np.isfinite(fstHits)
	                         *(np.abs(fstHits-tcore)<6000)   #max DOMpair length: 1557m
							 *(PEmap>PEthres))
	DOMfstHit = fstHits[tuple(finiteDOMs.T)]
	timeorder = np.argsort(DOMfstHit)
	finiteDOMs = finiteDOMs[timeorder]   #reorder finiteDOMs, earliest to latest
	DOMfstHit = DOMfstHit[timeorder]
	DOMpos = DOMcoord[tuple(finiteDOMs.T)]
	cDOMfstHitdelta = c*(DOMfstHit-tcore)
	DOMPEs = PEmap[tuple(finiteDOMs.T)]
	sumLogPEs = np.sum(np.log10(1+DOMPEs[np.linalg.norm(DOMpos-MSDpos, axis=1)<500]))   #a good gauge of the size of the event
	
	NPIX = hp.nside2npix(NSIDE)   #number of sky pixels
	skypixels = np.full((NPIX, 13), np.nan)
	DOMsPixeldict = {}
	for i in range(NPIX):
		axes_prime = np.array(new_axes(*hp.pix2ang(NSIDE, i)[::-1]))
		MSDposp, DOMposp = np.dot(MSDpos, axes_prime.T), np.dot(DOMpos, axes_prime.T)
		
		#collect DOMs triggering inside the moving cylindar---------------------------------------
		DOMsPixel = np.flatnonzero((np.linalg.norm(DOMposp[:,:2]-MSDposp[:2], axis=1) < cylinder[0]) \
		                            *(np.abs(DOMposp[:,2]-MSDposp[2]-cDOMfstHitdelta) < cylinder[1]))
		if(len(DOMsPixel)<0.1*len(DOMpos)  or len(DOMsPixel)==0): continue   #mainly to speed up the loop
		#-----------------------------------------------------------------------------------------
		
		#examine trackchain, look for weaknesses in gaps -- if too large, cut out
		zpPixel = DOMposp[:,2][DOMsPixel]   #zprime of every DOM in the DOMsPixel
		order = np.argsort(zpPixel)
		zpPixel, DOMsPixel = zpPixel[order], DOMsPixel[order]   #ordered by zprime
		sp = np.argmin(np.abs(zpPixel-MSDposp[2]))   #splitpoint, between pre and post cascade
		gaps1, gaps2 = np.diff(zpPixel[:sp+1]), np.diff(zpPixel[sp:])
		gaps1_rules = max_deltzp*np.clip(np.arange(1,len(gaps1)+1), None, 4)
		gaps2_rules = max_deltzp*np.clip(np.arange(1,len(gaps2)+1)[::-1], None, 4)
		wps1, wps2 = np.flatnonzero(gaps1>gaps1_rules), np.flatnonzero(gaps2>gaps2_rules)   #weakpoints
		if(axes_prime[2][2]<cosz_thres):   #pardon weakpoints in dustlayer for zenith less than 82degrees
			wps1 = [p for p in wps1 if nondust_frac(DOMpos[DOMsPixel[p]][2], DOMpos[DOMsPixel[p+1]][2], dustlayer)*gaps1[p]>gaps1_rules[p]]
			wps2 = [p for p in wps2 if nondust_frac(DOMpos[DOMsPixel[sp+p]][2], DOMpos[DOMsPixel[sp+p+1]][2], dustlayer)*gaps2[p]>gaps2_rules[p]]
		cutpoint1 = max(wps1)+1 if(len(wps1)) else None
		cutpoint2 = min(wps2)+1+len(gaps1) if(len(wps2)) else None
		DOMsPixel = DOMsPixel[cutpoint1:cutpoint2]   #cut
		if(len(DOMsPixel)<4): continue
		zpPixel = zpPixel[cutpoint1:cutpoint2]       #cut
		
		#check light speed fit
		tPixel = DOMfstHit[DOMsPixel]-np.linalg.norm(DOMposp[:,:2][DOMsPixel]-MSDposp[:2], axis=1)/0.34   #c/tan(41)
		polyfit = np.polyfit(tPixel, zpPixel, 1)
		if(polyfit[0]<cmin): continue
		
		#determine burdenDOMs and vetoDOMs
		DOMcoordp = np.dot(DOMcoord, axes_prime.T)
		with np.errstate(invalid='ignore'):
			burdenZps = DOMcoordp[:,:,2][(np.linalg.norm(DOMcoordp[:,:,:2]-MSDposp[:2], axis=2) < cylinder[0])*(clearDOMs)]
		if(burdenZps.size<5): continue   #this 5 is kind of arbitrary
		burdenZps = np.sort(burdenZps)
		m,b = np.polyfit(burdenZps, np.arange(1, burdenZps.size+1), 1)
		preEdge, postEdge = max((1-b)/m, burdenZps[0]), min((burdenZps.size-b)/m, burdenZps[-1])
		preVoid, postVoid = max(zpPixel[0]-preEdge, 0), max(postEdge-zpPixel[-1], 0)
		vetoDOMs = finiteDOMs[DOMsPixel[(zpPixel<MSDposp[2]-outdefn)*(tPixel<tcore)]]
		vetoDOMsPEs = np.array([linear_read(tcore, *tuple(DOMdict[d[0]][d[1]].T)) for d in vetoDOMs])
		preburden = np.sum(burdenZps<MSDposp[2]-outdefn)   #number of burdenDOMs up to the cascade boundary
		
		#record the hotpixel
		skypixels[i] = i, zpPixel[-1]-zpPixel[0], preVoid, postVoid, preburden, polyfit[0]/c-1, \
					   len(DOMsPixel), np.sum(DOMPEs[DOMsPixel]), np.sum(np.log10(1+DOMPEs[DOMsPixel])), \
		               len(vetoDOMs), np.sum(vetoDOMsPEs), np.sum(np.log10(1+vetoDOMsPEs)), np.sum(np.abs(zpPixel-np.median(zpPixel)))
		DOMsPixeldict[i] = DOMsPixel   #save selected DOMs in each pixel for linefit later
					   
	
	#score: 4:through-going track, 3:stopping track, 2:starting track, 1:cascade, 0:no hotpixel
	keys = {'central':0, 'longest':1, 'prevoid':2, 'postvoid':3, 'preburden':4, 'presumLogPEs':5, 'pathlength':6}
	if(not np.any(np.isfinite(skypixels[:,0]))): return np.full((7, skypixels.shape[1]), -1), sumLogPEs, np.full(3,np.nan), keys
	
	#prefull pixels are suspects
	with np.errstate(invalid='ignore'): prefull = skypixels[:,2]<thresVoid
	pathlength_pixel = int(skypixels[:,0][prefull][np.argmin(np.sum(skypixels[:,1:4][prefull], axis=1))]) if(np.any(prefull)) \
	                                                                                                      else np.nanargmin(skypixels[:,2])
	#find the best pixel based on sumlogPEsPixel - the best pixel has maximum 'intensity'
	with np.errstate(invalid='ignore'):
		cut_mask = (np.abs(skypixels[:,5])<=np.nanquantile(np.abs(skypixels[:,5]), 0.25))   #fit speed cut
		cut_mask *= (skypixels[:,1]>=np.nanquantile(skypixels[:,1][cut_mask], 0.5))         #tracklength cut
		cut_mask *= (skypixels[:,8]>=np.nanquantile(skypixels[:,8][cut_mask], 0.5))         #sumLogPEs cut
	candidate_pixels = np.flatnonzero(cut_mask)
	trackness = np.array([np.mean(np.nan_to_num(skypixels[:,12][table_smoothing[pix]])) for pix in candidate_pixels])
	top_pixels = candidate_pixels[trackness>0.97*trackness.max()]
	top_dirs = np.array(hp.pix2vec(NSIDE, top_pixels)).T
	angdists = [np.arccos(np.clip(np.sum(np.full(top_dirs.shape, vec)*top_dirs, axis=1),-1,1)) for vec in top_dirs]
	central_pixel = top_pixels[np.argmin(np.median(angdists, axis=1))]   #choose the most centered of the top pixels
	hotpixels = skypixels[hp.query_disc(NSIDE, hp.pix2vec(NSIDE, central_pixel), 0.349)]   #20degrees disc
	key_pixels = [central_pixel, 
	              int(hotpixels[:,0][np.nanargmax(hotpixels[:,1])]), 
	              int(hotpixels[:,0][np.nanargmin(hotpixels[:,2])]), 
	              int(hotpixels[:,0][np.nanargmin(hotpixels[:,3])]), 
	              int(hotpixels[:,0][np.nanargmin(hotpixels[:,4])]), 
	              int(hotpixels[:,0][np.nanargmax(hotpixels[:,11])]), 
	              pathlength_pixel]   #prefull pixel of the shortest pathlength
	
	#plain linefit on DOMs in the central pixels
	DOMsPixel = DOMsPixeldict[central_pixel]
	axes_prime = np.array(new_axes(*hp.pix2ang(NSIDE, central_pixel)[::-1]))
	DOMposp, MSDposp = np.dot(DOMpos, axes_prime.T), np.dot(MSDpos, axes_prime.T)
	tPixel = DOMfstHit[DOMsPixel]-np.linalg.norm(DOMposp[:,:2][DOMsPixel]-MSDposp[:2], axis=1)/0.34   #c/tan(41)
	linefitdir = np.polyfit(tPixel, DOMpos[DOMsPixel],1)[0]
	linefitdir *= 1/np.linalg.norm(linefitdir)
	
	return skypixels[key_pixels], sumLogPEs, linefitdir, keys


#create a NPIX*NPIX table of boolean values representing every pixel and its nearby pixels (within a disc of specified angular distance)
def area_table_build(NSIDE=16, radius=np.radians(20)):
	NPIX = hp.nside2npix(NSIDE)
	area_table = np.full((NPIX, NPIX), False)
	for i in range(NPIX): area_table[i][hp.query_disc(NSIDE, np.array(hp.pix2vec(NSIDE, i)).T, radius)] = True
	return area_table
#same purpose as area_table_build(), except return a dictionary of numpy arrays of skypixel indices
def area_dict_build(NSIDE=16, radius=np.radians(20)):
	return {i: hp.query_disc(NSIDE, np.array(hp.pix2vec(NSIDE, i)).T, radius) for i in range(hp.nside2npix(NSIDE))}
	

#clean fstHit_raw - remove a coincident track, so track_sweeper can work better
def track_cleaner(fstHit_raw, PEmap, DOMcoord, MSS, MSD, 
                  NSIDE=16, thresVoid=150, xyrad=150, ztsize=170, threstracklength=500, PEthres=0.5, c=0.299792458):
	outdefn = 170
	MSDboundary = 300
	holesizethres = 3
	numDOMthres = 5
	MSDpos = DOMcoord[MSS, MSD]   #best estimate of cascade's location
	with np.errstate(invalid='ignore'):
		finiteDOMs = np.argwhere(np.isfinite(fstHit_raw)
	                             *(PEmap>PEthres)
	                             *(np.linalg.norm(DOMcoord-MSDpos, axis=2)>outdefn))
	DOMfstHit = fstHit_raw[tuple(finiteDOMs.T)]
	timeorder = np.argsort(DOMfstHit)
	finiteDOMs = finiteDOMs[timeorder]   #reorder finiteDOMs, earliest to latest, ensuring cdelta_t is always positive
	DOMfstHit = DOMfstHit[timeorder]
	DOMpos = DOMcoord[tuple(finiteDOMs.T)]

	#pair up DOMs
	numDOMs = len(finiteDOMs)
	pairs = pair_generator(numDOMs)   #all DOMpairs, DOM indices derived from finiteDOMs
	cdelta_t = c*(DOMfstHit[pairs[:,1]]-DOMfstHit[pairs[:,0]])
	delta_r = DOMpos[pairs[:,1]]-DOMpos[pairs[:,0]]
	delta_r_mag = np.linalg.norm(delta_r, axis=1)
	pairdir = (delta_r.T/delta_r_mag).T   #cartesian

	#Tabulate valid DOMpairs at every pixel in the sky
	F = cdelta_t/delta_r_mag   #the central piece of this method
	theta_min, theta_max = track_theta_bounds(F, g=50./delta_r_mag, tanc=0.8692867378162267)
	NPIX = hp.nside2npix(NSIDE)   #number of sky pixels
	skypixels_pairs = np.full((NPIX, len(pairs)), False)   #masks giving compatible DOMpairs at each pixel
	for i,vec in enumerate(pairdir):
		if(np.isnan(theta_max[i]) or abs(F[i]-1)>0.3): continue   #(can also be theta_min)
		inner_bound = hp.query_disc(nside=NSIDE, vec=vec, radius=theta_min[i])
		outer_bound = hp.query_disc(nside=NSIDE, vec=vec, radius=theta_max[i])
		skypixels_pairs[outer_bound, i] = True
		skypixels_pairs[inner_bound, i] = False
		
	trackness_max, coincidentDOMs = 0, np.array([])
	i0 = NSIDE*(6*NSIDE+2)   #NSIDE*(6*NSIDE+2):start of downgoing pixels
	for i, mask in enumerate(skypixels_pairs[i0:], i0):
		DOMsPixel = np.unique(pairs[mask])   #at each pixel, a subset of DOMs compatible with the pixel
		if(len(DOMsPixel)<numDOMthres): continue
		axes_prime = np.array(new_axes(*hp.pix2ang(NSIDE, i)[::-1]))
		DOMsPixelposp, MSDposp = np.dot(DOMpos[DOMsPixel], axes_prime.T), np.dot(MSDpos, axes_prime.T)
		MSDcut = np.linalg.norm(DOMsPixelposp[:,:2]-MSDposp[:2], axis=1)>MSDboundary
		DOMsPixel, DOMsPixelposp = DOMsPixel[MSDcut], DOMsPixelposp[MSDcut]
		if(np.sum(MSDcut)<numDOMthres or (DOMsPixelposp[:,2].max()-DOMsPixelposp[:,2].min())<threstracklength): continue
		
		#select the optimal zt window by moving it around to grab the maximum number of DOMsPixel
		ztpos = DOMsPixelposp[:,2]-c*DOMfstHit[DOMsPixel]
		ztpos2 = np.sort(ztpos)
		counts = np.arange(1, len(ztpos2)+1)
		samples = np.linspace(ztpos2.min(), ztpos2.max()-ztsize, 30)
		optimum = np.argmax([linear_read(x+ztsize, ztpos2, counts)-linear_read(x-ztsize, ztpos2, counts) for x in samples])
		cleanmask = np.abs(ztpos-(samples[optimum]))<=ztsize
		cleanmask*=np.linalg.norm(DOMsPixelposp[:,:2]-np.median(DOMsPixelposp[:,:2][cleanmask], axis=0), axis=1)<xyrad
		if(np.sum(cleanmask)<numDOMthres): continue
		zpPixel = np.sort(DOMsPixelposp[:,2][cleanmask])
		zpmin, zpmax = zpPixel.min(), zpPixel.max()
		if((zpmax-zpmin)<max(threstracklength, holesizethres*np.diff(zpPixel).max())): continue
		
		xycenter = np.median(DOMsPixelposp[:,:2][cleanmask], axis=0)
		DOMcoordp = np.dot(DOMcoord, axes_prime.T)
		with np.errstate(invalid='ignore'): burdenZps = DOMcoordp[:,:,2][np.linalg.norm(DOMcoordp[:,:,:2]-xycenter, axis=2) < xyrad]
		preEdge, postEdge = (np.min(burdenZps), np.max(burdenZps)) if(len(burdenZps)) else (np.nan, np.nan)
		preVoid, postVoid = zpmin-preEdge, postEdge-zpmax
		if(preVoid<thresVoid and postVoid<thresVoid):
			trackness = np.sum(np.abs(zpPixel-np.median(zpPixel)))
			if(trackness>trackness_max):
				trackness_max = trackness
				coincidentDOMs = DOMsPixel[cleanmask]
	
	return finiteDOMs[coincidentDOMs] if(len(coincidentDOMs)) else coincidentDOMs




