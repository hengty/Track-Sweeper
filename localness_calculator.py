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