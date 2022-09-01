#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/icetray-start
#METAPROJECT /home/bty/combo/build

#search for large cascades with a small entering track
#run with python hease.py [sim] [run] [subrun]
#ty@wisc.edu
#Last update: 04 Jan 2022

#---PARAMETERS----------------------------------------------------------------------------
localness_thres = 0.5
additionalPEs_thres = 2500
MSS_thres = 1.5   #threshold for not selecting the brightest string as the MSS
MSD_bound = (5, 60)   #DOM number cut
PEthres = 0.5   #for a DOM to be considered, in TrackSweeper
nside = 16   #setting the number of healpix pixels
cylinder = (160., 240.)   #size of the search cylindar in track_sweeper
radius_smoothing = 10   #degree
minVoid = 150.   #min length(m) from detector's edge for the track to be considered "full" -- a through going track
tcscd_thres = 0.1   #tcscd set at when 10% of DOMs in neighborhood have triggered
scriptdir = '/home/bty/hease/nov092021'   #location of BTyFunctions.py and spice_3.2.2
#-----------------------------------------------------------------------------------------

#---SETUP---------------------------------------------------------------------------------
import sys
sys.path.append(scriptdir)
from BTyFunctions import *

#dat or sim
if(os.sys.argv[1]=='dat'):
	year, date, run, subrun = os.sys.argv[2:]
	rundir = '/data/exp/IceCube/'+year+'/filtered/level2/'+date+'/Run'+run
	savedir = scriptdir+'/outputs/filtered'+run+'_'+subrun[-15:-7]
	for item in os.listdir(rundir):   #find GCD dir
		if('_GCD.i3' in item):
			GCDdir = rundir+'/'+item
			break
elif(os.sys.argv[1]=='sim'):
	path, run, subrun = os.sys.argv[2:]
	#rundir = '/data/sim/IceCube/2015/filtered/level2/CORSIKA-in-ice/12604/'+run
	rundir = path+'/'+run
	savedir = scriptdir+'/outputs/sim/filtered'+path[-2:]+'_'+run+'_'+subrun[-13:-7]#scriptdir+'/outputs/sim/filtered'+path[-5:]+'_'+run+'_'+subrun[-13:-7]
	for item in os.listdir(rundir):   #find GCD dir
		if('GeoCalibDetectorStatus' in item):
			GCDdir = rundir+'/'+item
			break
		GCDdir = '/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2016.57531_V0.i3.gz'
else: raise Exception('Need to specify dat or sim')

#maps revealing structures of the detector that is IceCube
neighbor_map, territory_map,        \
boundary_strings, boundaries,       \
deep_strings, I3Geometry, DOMcoord, \
xybounds, zbounds = maps_generator(GCDdir)
scatmap, scat_len = icemodel(scriptdir+'/spice_3.2.2', DOMcoord)[0:2]
table_smoothing = area_dict_build(NSIDE=nside, radius=np.radians(radius_smoothing))
#table_hot       = area_dict_build(NSIDE=nside, radius=np.radians(radius_hot))
with np.errstate(invalid='ignore'): clearDOMs = (scatmap>20)*((DOMcoord[:,:,2]<-188)+(DOMcoord[:,:,2]>16))   #burdenDOMs
#-----------------------------------------------------------------------------------------


#------------------------------------------------------------------------START------------------------------------------------------------------------#
f = dataio.I3File(rundir+'/'+subrun, 'r')
g = [] 
while(True):
	try: q = f.pop_daq()
	except: break
	data = q['InIcePulses'].apply(q)

	if(os.sys.argv[1]=='sim'):
		if(not 'MCPrimary' in q.keys()): weighting.get_weighted_primary(q)   #this creates 'MCPrimary' in the frame
		if(q['MCPrimary'].energy<20000): continue

	#localness
	localness, ballooness, totalPEs, MSS, MSD, PEmap = localness_calculator(data, DOMcoord[:,:,2], deep_strings, neighbor_map, MSS_thres=MSS_thres)
	if(localness<localness_thres or \
	   totalPEs*(1-ballooness)<additionalPEs_thres or \
	   MSS in boundary_strings or not MSD_bound[0]<MSD<MSD_bound[1]): continue
	#----------------------------------------------------------------cut----------------------------------------------------------------#

	#DOMdict, stringdict, fstHit
	DOMdict, stringdict, charging_curve = build_I3dict(data, deep_strings)
	fstHit_raw = np.full((79, 61), np.inf)    #string, DOM -- ignore DeepCore
	for string in DOMdict:
		for DOM in DOMdict[string]:
			fstHit_raw[string][DOM] = DOMdict[string][DOM][0][0]
	stringlist = [MSS] + list(neighbor_map[MSS][:,0].astype('int'))
	#posxy = xyEstimate(stringlist, np.sum(PEmap[stringlist], axis=1), DOMcoord[stringlist, MSD][:,:2])
	#posz = zfitter2(np.argmax(PEmap[MSS]), DOMcoord[MSS,:,2], PEmap[MSS], 5)[0]

	#---track sweeper----------------------------------------------------------------------------------------------
	tcscd = np.quantile(fstHit_raw[stringlist, max(MSD-10,1):min(MSD+9,60)+1], tcscd_thres)   #cascade time estimate
	TS0 = track_sweeper(fstHit_raw, tcscd, PEmap, DOMcoord, MSS, MSD, boundary_strings, DOMdict, 
	                    table_smoothing, clearDOMs, NSIDE=nside, cylinder=cylinder, PEthres=PEthres, thresVoid=minVoid)
	TS1 = track_sweeper(fstHit_raw, fstHit_raw[MSS,MSD], PEmap, DOMcoord, MSS, MSD, boundary_strings, DOMdict, 
	                    table_smoothing, clearDOMs, NSIDE=nside, cylinder=cylinder, PEthres=PEthres, thresVoid=minVoid)
	TS = TS1 if (TS1[0][5][11]>=TS0[0][5][11] or not np.isfinite(TS0[2][0])) else TS0

	sumLogPEs = TS[1]	
	tracklength_max = np.nanmax([TS0[0][1][1], TS1[0][1][1]])
	#pathlength = np.sum(TS[0][1][1:4])
	preVoidmin = np.nanmin([TS0[0][2][2], TS1[0][2][2]])
	postVoidmin = np.nanmin([TS0[0][3][3], TS1[0][3][3]])
	preBurden_min = np.nanmin([TS0[0][4][4], TS1[0][4][4]])
	numVetoDOMs, presumPEs_max, presumLogPEs_max = np.nanmax([TS0[0][5][9:12], TS1[0][5][9:12]], axis=0)
	prefull_pathlength_min = np.nanmin([np.sum(TS0[0][6][1:4]) if(TS0[0][6][2]<minVoid) else 2000., 
	                                    np.sum(TS1[0][6][1:4]) if(TS1[0][6][2]<minVoid) else 2000.])
	pixlnum = TS[0][0][0]
	finalz = TS[2][2]
	#--------------------------------------------------------------------------------------------------------------
	
	#find the main physics frame of the event, save linefit
	p = f.pop_frame()
	while(p['I3EventHeader'].sub_event_stream!='InIceSplit'):   #pop_frame() untill reach 'InIceSplit'
		p = f.pop_frame()
		if(p['I3EventHeader'].event_id!=p['I3EventHeader'].event_id): raise ValueError('p frame error')
	linefit_dir = [p['PoleMuonLinefit'].dir.x, p['PoleMuonLinefit'].dir.y, p['PoleMuonLinefit'].dir.z] if 'PoleMuonLinefit' in p else [np.nan]*3	
	
	#this part for data only
	if(os.sys.argv[1]=='dat'):
		g.append([q['I3EventHeader'].run_id, int(subrun[-15:-7]), q['I3EventHeader'].event_id, totalPEs, sumLogPEs, localness, ballooness, presumPEs_max, 
                          presumLogPEs_max, preBurden_min, numVetoDOMs, pixlnum, tracklength_max, finalz, prefull_pathlength_min, preVoidmin, postVoidmin] + linefit_dir)
		continue
		

		
	#---extract MC information (this part for sim only)--------------------------------------------------------------------------
	#direction, prelengthMC, preEdeposited, Estart. for comparing to the numVetoDOMs, prelengthh, and presumlogPEs_max
	from icecube import weighting
	if(not 'MCPrimary' in q.keys()): weighting.get_weighted_primary(q)   #this creates 'MCPrimary' in the frame
	tree = 'I3MCTree' if('I3MCTree' in q) else 'I3MCTree_preMuonProp'
	primary = q['MCPrimary']#q['I3MCTree'].get_head()
	axes_prime = np.array(new_axes(primary.dir.azimuth, np.pi+primary.dir.zenith))
	bound_coordprime = np.dot(DOMcoord[tuple(boundaries['all'].T)], axes_prime.T)
	boundDOMsInside = np.linalg.norm((bound_coordprime[:,:2]-np.dot(primary.pos, axes_prime.T)[:2]), axis=1)<160   #within 160m radius
	preZp_start = np.min(bound_coordprime[boundDOMsInside][:,2])
	if('MMCTrackList' in q.keys()):
		points, energies = track_energies(q[tree], q['MMCTrackList'])
		nstop = max(np.argmin(np.diff(energies))-10, 0)   #index of where to read energy before the big cascade
		preZp_stop = np.dot(points[nstop], axes_prime[2])
		Estart = linear_read(preZp_start, np.dot(points, axes_prime[2]), energies)   #energy of track entering the detector
		Estop = energies[nstop]
		prelengthMC = preZp_stop-preZp_start
		preEdeposited = Estart-Estop
		totalweight = q['I3MCWeightDict']['TotalWeight'] if('I3MCWeightDict' in q) else -1
		oneweightnfiles = q['I3MCWeightDict']['OneWeight']/(0.5*q['I3MCWeightDict']['NEvents']) if('I3MCWeightDict' in q) else -1
	else:
		Estart, preEdeposited, prelengthMC, nstop, totalweight, oneweightnfiles = np.full(6, -1)
		points = np.full((10,3), np.nan)
	#----------------------------------------------------------------------------------------------------------------------------
	g.append([q['I3EventHeader'].run_id, int(subrun[-13:-7]), q['I3EventHeader'].event_id, totalPEs, sumLogPEs, localness, ballooness, presumPEs_max, 
                  presumLogPEs_max, preBurden_min, numVetoDOMs, pixlnum, tracklength_max, finalz, prefull_pathlength_min, preVoidmin, postVoidmin] + linefit_dir + \
                  [primary.dir.azimuth, primary.dir.zenith, Estart, preEdeposited, prelengthMC]+list(points[nstop+10])+[primary.energy, totalweight, oneweightnfiles])

np.save(savedir, g)
f.close()	













