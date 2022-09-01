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