"""
It is collection of programs since 2006. 
Partly written by me, and partly collected from forums, etc. 
This is the main module I have and load on nearly any of my Python programs.

to use it, you should first import it:

import reza

and then use the moduel functions

print(reza.stddev(x))
"""
######################################
# import necessary external packages #
######################################
import numpy, scipy
from pylab import *
from numpy import *
from numpy.fft import *
from scipy import ndimage

#######################################################
# shift                                               #
#######################################################
def cshift(l, offset):
  offset %= len(l)
  return numpy.concatenate((l[-offset:], l[:-offset]))

def shift1d(l, offset):
  offset %= len(l)
  return numpy.concatenate((l[-offset:]))

#######################################################
# shift  in 2D                                        #
#######################################################
def shift2d(image, offset):
  scipy.ndimage.interpolation.shift(image, (offset[0], offset[1]))
#######################################################
# shift  in 1D                                        #
#######################################################
def shift(array, key):
    return numpy.roll(array, key)
#######################################################
# paused                                              #
#######################################################
def paused():
        p_enetr = raw_input("Press Enter to continue...")
        return
#######################################################
# 1 sigma standard deviation                          #
#######################################################
def stddev(num):
	return numpy.std(num)
#######################################################
# smooth in 1D                                        #
#######################################################
def smooth(x,window_len,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string

    NOTE: length(output) != length(input), to correct this: 
          return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

#######################################################
# smooth in 1D                                        #
#######################################################
def gauss_kern1(v, size):
    """ Returns a normalized 1D gauss kernel array for convolutions """
    size = int(size)
    tay = len(v)/3
    x = mgrid[-tay:tay+1]
    g = exp(-(x**2/float(size)))
    return g / g.sum()
#######################################################
# Gaussian blur  in 1D                                #
#######################################################
def blur_vector(im, n) :
    import scipy.signal
    """ It blurs the 1D array by convolving with a gaussian kernel 
        of typical size n. 
    """
    g = gauss_kern1(im, n)
    print('the prog ', len(g), len(im))
    improc = scipy.signal.convolve(im,g, mode='same')
    print(len(improc))
    #improc = shift(improc, n)
    print(len(improc))
    return(improc)


#######################################################
# Gaussian kernel for smoothing in 2D                 #
#######################################################
def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = mgrid[-size:size+1, -sizey:sizey+1]
    g = exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

#######################################################
# Gaussian blur  in 2D                                #
#######################################################
def blur_image(im, n, ny=None):
    import scipy.signal
    """ It blurs the image by convolving with a gaussian kernel 
        of typical size n. The optional keyword argument ny 
        allows for a different size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = scipy.signal.convolve(im,g, mode='same')
    return(improc)

#######################################################
# average2bins                                        #
#######################################################
def average2bins(x, y, n):
    """ It shrinks the vector size by a factor n.
    """
    g = numpy.round(len(x)/n)
    new_x = numpy.zeros(g); new_y = numpy.zeros(g)
    # sort in x
    sort_index = numpy.argsort(x)
    x_tmp = x[sort_index]
    y_tmp = y[sort_index]
    for i in range(0,g):
      new_x[i] = numpy.mean(x_tmp[i*n:((i+1)*n-1)])
      new_y[i] = numpy.mean(y_tmp[i*n:((i+1)*n-1)])
    r = [new_x, new_y]  
    return r
#######################################################
# sigma_clip                                          #
#######################################################
def sigma_clip(data, low, high):
    """ It performs sigma clipping in inpout.
    """
    g = numpy.median(data)
    s1 = g - numpy.std(data) * low
    s2 = g + numpy.std(data) * high
    data[where(data < s1)] = s1
    data[where(data > s2)] = s2
    return data

#######################################################
# median_clip                                         #
#######################################################
def median_clip(data):
    """ It removes CRs and hot pixels.
    """
    g = blur_image(data, 3)
    x = abs(data - g)
    s1 = numpy.std(g) * 1.
    q = where(x > s1)
    print('sssssss', len(q))
    data[q] = g[q]
    return data

#######################################################
# list the immediate child directories                #
#######################################################
def ls_dir(fpath):
    import os
    exist = next(os.walk(fpath))[1]
    return exist
#######################################################
# list all sub directoreis                            #
#######################################################
def ls_dir_all(fpath):
    import os
    exist = [x[0] for x in os.walk(fpath)]
    return exist
#######################################################
# check existence of a file                           #
#######################################################
def check_existence(filename):
    import os.path
    exist = os.path.exists(filename)
    return exist
#######################################################
# number of a file in a directory                     #
#######################################################
def ls(fpath):
    import os, os.path
    exist = len([name for name in os.listdir(fpath) if os.path.isfile(fpath+name)])
    return exist
#######################################################
# #file in a directory with a given exttension        #
#######################################################
def lsd(fpath, ext):
    # ext , e.g., '*.py'    
    import glob
    exist = len(glob.glob1(fpath,ext))
    return exist
#######################################################
# find minimum using a Parabola (2nd order) fit       #
# result should be comparable with lpff routine below #
#######################################################
def min_parabola(xx, x):
    import warnings
    warnings.simplefilter('ignore', np.RankWarning)
    n = len(x)
    #xx = numpy.arange(0,n)
    res = numpy.polyfit(xx, x, 2)
    output =  -res[1]/(2*res[0])
    return output

#######################################################
# program to find a file                              #
#    find('*.avi', sys.argv[1])                       #
#######################################################
def find(pattern, path):
    import os, fnmatch, sys
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

#######################################################
# another program to find a file                      #
#######################################################
def find2(pattern, path):
    import os, glob, sys
    result = []
    current_dir = os.getcwd()
    os.chdir(path)
    for filea in glob.glob(pattern):
      result.append(filea)
    
    os.chdir(current_dir)
    return result

#######################################################
# lpff: find minimum of a line profile                #
# result should be comparable to min_parabola (above) #
#######################################################
def lpff(arr):
    import warnings
    warnings.simplefilter('ignore', ComplexWarning)
    from numpy.fft import fft, ifft
    posmin = arr.argmin()
    k1 = (posmin - 7)
    if (k1 < 0):
        k1 = 0
    k2 = (posmin + 7)
    if (k2 > (len(arr)-1)):
        k2 = len(arr) - 1
    #print posmin, k1, k2
    arr = arr[k1:k2]
    sz = numpy.array(arr.shape)
    mid = sz / 2.
    tpi = 2. * numpy.pi
    dp = 360. / sz                   #Grad/pixel
    l1 = arr
    fl1 = numpy.fft.ifft(l1)
    lp = - numpy.arctan( numpy.imag(fl1[1]) / float(fl1[1]) )/tpi*360.
    pos= lp/dp + mid
    return pos

#######################################################
# Computes Pearson's r correlation between sets (x,y) #
#######################################################
def pearson(x,y):
    
    nx=x.size()
    ny=y.size()
    if (nx != ny):
        print("(Pearson) ARRAY SHOULD HAVE SAME SIZE!")
        
    
    d=(nx-1.)*standard_deviation(x)*standard_deviation(y)
    r=sum(x-mean(x))*sum(y-mean(y))/d

    return r

#######################################################
# rebin by factor                                     #
#######################################################
def rebin(a, shape):
    print(shape[0], shape[1])
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    print(sh)
    return a.reshape(shape).mean(-1).mean(1)
##################################
# power spectrum of a 1D profile #
##################################
def powspec(profile):

    nt= size(profile)
    #- sampling of profile in "pixel" space
    tstep=1.
    #- array definition in "frequency" space
    f = arange(0.,1./tstep,1./tstep/nt)
    sf= fft(profile)
    
    subplot(211)
    plot(arange(nt), profile)
    xlabel('wavelength (pixel)')
    ylabel('ADUs')
    subplot(212)
    plot(f[0:nt/2],abs(sf[0:nt/2]))
    xlabel('frequency (pixel^-1)')
    ylabel('power spectrum')
    
    show()

############################################
# plots 4 profiles on same frame (I,Q,U,V) #
############################################
def tvscl(im):

    imshow(im, interpolation='nearest')
    show()

############################################
# plots 4 profiles on same frame (I,Q,U,V) #
############################################
def plot4(p1,p2,p3,p4):

    subplot(221)
    plot(p1)
    
    subplot(222)
    plot(p2)
    
    subplot(223)
    plot(p3)
    
    subplot(224)
    plot(p4)
    
    show()

##################################
# plots 2 profiles on same frame #
##################################
def plot2(p1,p2):

    subplot(211)
    plot(p1)
    
    subplot(212)
    plot(p2)

    show()

##########################################
# plots 4 images on same frame (I,Q,U,V) #
##########################################
def im4(im1,im2,im3,im4):

    subplot(221)
    imshow(im1) #,interpolation='nearest')
    title(r'no 1')
    subplot(222)
    imshow(im2) #,interpolation='nearest')
    title(r'no 2')
    subplot(223)
    imshow(im3) #,interpolation='nearest')
    title(r'no 3')
    subplot(224)
    imshow(im4) #,interpolation='nearest')
    title(r'no 4')
    show()

################################
# plots 2 images on same frame #
################################
def im2(im1,im2):

    subplot(121)
    imshow(im1,interpolation='nearest')
    
    subplot(122)
    imshow(im2,interpolation='nearest')
    
    show()


#-------------------------------------------------
# prints statistical parameters of the input     -
#-------------------------------------------------
def statist(param):
    # prints min/max/mean/std of the input
    a = zeros((4), float)
    a[0] = numpy.min(param)
    a[1] = numpy.max(param)
    a[2] = scipy.mean(param)
    a[3] = scipy.std(param)
    print('min = ', a[0], '  max = ',a[1])
    print('mean = ',a[2], '  std = ',a[3])
    return a

#-------------------------------------------------
# creates a 1D array like Findgen in IDL    -
#-------------------------------------------------
def findgen1(num):

    # prints min/max/mean/std of the input
    a = zeros((num), float32)
    for i in arange(num):
       a[i] = i * 1.0
    return a
#-------------------------------------------------
# creates a 2D array like Findgen in IDL    -
#-------------------------------------------------
def findgen(num):

    # prints min/max/mean/std of the input
    a = zeros((num,num), float)
    for i in arange(num):
        for j in arange(num):
           a[i,j] = (i+j) * 1.0
    return a


#-------------------------------------------------
# this function show an image and                -
# prints x/y/value of the mouse position         -
#-------------------------------------------------
def tvwin(dat):

    imshow(dat, origin='lower', interpolation='nearest')
    naxis2 = dat.shape[0] 
    naxis1 = dat.shape[1] 

    def on_move(event):
        # get coordinates
        x, y = event.x, event.y
 
        if (event.inaxes)and(event.xdata < naxis1)and(event.ydata < naxis2):
            print('x=', int(event.xdata), ' y=',int(event.ydata))
            xi = int(around(event.xdata))
            yi = int(around(event.ydata))
            print(dat[naxis2-yi, xi])

    connect('motion_notify_event',on_move)
    show()
 
#-------------------------------------------------
# this function show an image and  plot profile  -
# corresponding to the mouse position            -
#-------------------------------------------------
def tvwinp(dat):

    subplot(131)
    ion()
    isinteractive()
    imshow(dat, origin='lower', interpolation='nearest')
    draw()
    dat.shape
    naxis2 = dat.shape[0] 
    naxis1 = dat.shape[1] 

    def on_move(event):
        # get coordinates
        x, y = event.x, event.y
 
        if (event.inaxes): #and(event.xdata < naxis1)and(event.ydata < naxis2):
            #print 'x=', int(event.xdata), ' y=',int(event.ydata)
            xi = int(around(event.xdata))
            yi = int(around(event.ydata))
            print('x=', int(event.xdata), ' y=',int(event.ydata), ' data =',dat[naxis2-yi, xi])
            ky = naxis2-int(event.ydata) 
            kx = int(event.xdata)
            ax = zeros((naxis1), float) + kx
            ay = zeros((naxis2), float) + ky
            subplot(132)
            plot(dat[ky,:], hold=False)
            title('horizontal')
            subplot(133)
            plot(dat[:,kx], hold=False)
            title('vertical')
            #plot(dat[ky,:],'b', dat[:,kx], 'r', hold=False)
            draw()

    connect('button_press_event',on_move)
    ioff()
    show()

#-------------------------------------------------
# this function show an image and  read two      -
# mouse clicks and finally draw a rectangle      -
# for the selected region                        -
#-------------------------------------------------
def zoom_box(dat):
    import matplotlib, time
    from matplotlib import verbose, ticker, patches
    from matplotlib.patches import Rectangle
    from scipy import loadtxt, savetxt
    import os, sys
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #subplot(111)
    ion()
    isinteractive()
    imshow(dat, origin='lower',cmap=cm.gray, interpolation='nearest')
    draw()
    dat.shape
    naxis2 = dat.shape[0] 
    naxis1 = dat.shape[1] 
    print('click the lower left and upper right')
    posx=[]
    posy=[]
    # create  a temp file
    if check_existence('zoom_box_coords.txt'):
        os.system('rm  zoom_box_coords.txt')
    
    f_handle = file('zoom_box_coords.txt', 'a')
    def on_move(event):
        # get coordinates
        x, y = event.x, event.y
        
        if (event.inaxes): #and(event.xdata < naxis1)and(event.ydata < naxis2):
            kx = int(around(event.xdata))
            ky = int(around(event.ydata))
            draw()
            posx.append(kx)
            posy.append(ky)
            snew = [kx, ky]
            savetxt(f_handle, snew, fmt='%5.0i')
            if (len(posx) == 2):
                 dx = posx[1] - posx[0]
                 dy = posy[1] - posy[0]
                 rects = [Rectangle(xy=[posx[0], posy[0]], width=dx, height=dy,angle=0,fill=False,color='r')]
                 ax.add_artist(rects[0])
            time.sleep(0.5)
          
    connect('button_press_event',on_move)
    ioff()
    show()
    f_handle.close()
    b = loadtxt('zoom_box_coords.txt', dtype='int')
    #print(b)
    if check_existence('zoom_box_coords.txt'):
        os.system('rm  zoom_box_coords.txt')
    return b

#def read_two_clicks(data):
#    import time
#    imshow(data, origin='lower',cmap=cm.gray, interpolation='nearest')
#    import pymouse
#    mouse = pymouse.PyMouse()
#    mouse.screen_size()  # click  move position press release
#    #Click the mouse on a given x, y and button.
#    #Button is defined as 1 = left, 2 = right, 3 = middle.
#    
#    #state = mouse.click()
#    #if (state == 1):
#    show()
#    time.sleep(3)
#    x = mouse.position()
#    print(x)
    

#-------------------------------------------------
# this function calls the zoom_box to iteratvely -
# select corners of a ractangle                  -
#-------------------------------------------------
def my_zoom_box(input_map):
    #x = numpy.random.normal(0., 1., (600,600))
    #input_map = x
    modex = 0
    while (modex < 1):
      coords = zoom_box(input_map)
      #read_two_clicks(input_map)
      print('satisfied with the selection [0/1]?')
      raw_input('Enter your input:')
      try:
        modex=int(raw_input('Input:'))
      except ValueError:
        print("Not a number")
      #finally:
      #  print(modex)
    print('done.')
    print('x1= ', coords[0], 'y1= ', coords[1])
    print('x2= ', coords[2], 'y2= ', coords[3])
    return coords
    
#-------------------------------------------------
# user sends an image, interactively makes two   -
# clicks, and get the cut-out image as result    -
#-------------------------------------------------
def cut_out_img(input_data):
    s = my_zoom_box(input_data)
    print(s)
    new_data = input_data[s[1]:s[3], s[0]:s[2]]
    return [new_data, s]

#-------------------------------------------------
# this function show an image and  plot profile  -
# corresponding to the mouse position         -
#-------------------------------------------------
def select(dat):

    isinteractive()
    imshow(dat, origin='lower',cmap=cm.gray, interpolation='nearest')
    draw()
    p = dat.shape()
    naxis2 = p[1] #dat[:,0].size()
    naxis1 = p[2] #dat[0,:].size()

    def on_move(event):
        # get coordinates
        x, y = event.x, event.y
 
        if (event.inaxes): #and(event.xdata < naxis1)and(event.ydata < naxis2):
            #print 'x=', int(event.xdata), ' y=',int(event.ydata)
            xi = int(around(event.xdata))
            yi = int(around(event.ydata))
            print('x=', int(event.xdata), ' y=',int(event.ydata), ' data =',dat[naxis2-yi, xi])

#    coord=[xi, naxis2-yi]

    connect('button_press_event',on_move)
    show()

#    return coord




#----------------------------------------------------
# this function show an image and  returns profile  -
# corresponding to the two input position           -
#----------------------------------------------------
def profile(dat, x1,x2,y1,y2):

    ion()
    dat.shape
    naxis2 = dat.shape[0] 
    naxis1 = dat.shape[1] 
    x1 = float(x1)
    x2 = float(x2)
    y1 = float(y1)
    y2 = float(y2)
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    out = zeros(length)
    yar = zeros(length)
    xar = zeros(length)

    stepx = dx/length
    stepy = dy/length
    for i in arange(length): 
       yar[i] = (float(i) * stepy) + y1
       xar[i] = (float(i) * stepx) + x1
       out[i] = dat[yar[i],xar[i]]

    return out





def rotate(source, target, angle):

  #defining the rotation matrix that we all know and love
  #   [  cos(theta)    sin(theta)  ]
  #   [  -sin(theta)   cos(theta)  ]
  rma = cos(angle)
  rmb = sin(angle)
  rmc = -1 * rmb
  rmd = rma
  tkp = target.shape
  width = tkp[0]
  height = tkp[1]
  print(width, height)
  # rather than actually making a 2x2 array, I jREMOVEDt made 4 variables that will
  # be referenced to and I'll do the matrix operations manually

  #we're going to loop through the target image
  for x in range(0, width):
    for y in range(0, height):

      # denoting the top left pixel as (1,1) isn't real good for doing rotations
      # therefore, I'm going to redefine another coordinate system where the center of
      # the image is (0,0) and negative coordinates will be utilized.
      xcor = x - int(width/2)
      ycor = y - int(height/2)

      # now I'm going to take these coordinates and rotate them REMOVEDing
      # the rotational matrix. These new coordinates (sx, sy) will be 
      # REMOVEDed to determine which color to look for in the source picture
      sx = xcor * rma + ycor * rmb + width/2
      sy = xcor * rmc + ycor * rmd + height/2
      # the beauty of this method is that when you loop through the 
      # target image, you will know that there will always be a coresponding 
      # color on the source image. If I were to loop through the source image
      # coordinates, then I wouldn't be guaranteed a 1 to 1 corelation of pixels
      # and there'll probably be an even distribution of "holes" in the final image
      #print x, y, sy, sy


      # if the rotated coordinates goes off the bounds of the source image, 
      # then ignore this loop and go on to the next one. 
      if ((sx < 0) or (sx > width)) or ((sy < 0) or (sy > height)):
        pass
      else:
        target[x, y] = source[sx, sy]




#######################################################
# calculate position of line cores for image.
# Fit polynomial on course, shift all images to line
# core at slit height /4 .  
#######################################################
def deskew(icorr, ilo,ihi,firsty,lasty,y1,y2, order):
    import pylab
#;--------------------------------------------------
# get position of line core in Stokes I
# find minimum inside range
#--------------------------------------------------      
    print('deskew ....')
    print('deskew ....', ilo, ihi)
    print('deskew ....', firsty, lasty)
    
    pos_range = numpy.zeros((lasty-firsty+1), float)
    for k in numpy.arange(firsty,lasty+1):
        posmin = icorr[k, ilo:ihi].argmin()
        xx = numpy.arange(ihi - ilo)
        rm = min_parabola(xx, icorr[k, ilo:ihi])
        pos_range[k-firsty] = rm # posmin
        print(posmin)
    posa = numpy.zeros((lasty-firsty+1), float)
    #pylab.imshow(icorr, origin='lower')

#---------------------------------------------------------------
# get line core position with lpff.pro around minimum intensity
# calculate curvature only between firsty:lasty
#---------------------------------------------------------------
    for k in numpy.arange(firsty,lasty+1):
        posa[k-firsty] = lpff(icorr[k, ilo+pos_range[k-firsty]-5:ilo+pos_range[k-firsty]+5])
        posa[k-firsty] += (pos_range[k-firsty] - 5)

#--------------------------------------------------------------
# fit polynomial of 2nd order to position of core along slit
# extend curve to full data range y1:y2
#--------------------------------------------------------------
    xf = numpy.arange(1,lasty-firsty-1)
    range_corr = xf
    res = numpy.polyfit(xf, posa[1:lasty-firsty-1], order)
    print('deskew ....order=', order)
   
    if (order == 1):
        yy_corr = res[1] + res[0]*range_corr
    if (order == 2):
        yy_corr = res[2] + res[1]*range_corr + res[0]*(range_corr)**2
    if (order == 3):
        yy_corr = res[3] + res[2]*range_corr + res[1]*(range_corr)**2 + res[0]*(range_corr)**3
    pylab.plot(pos_range)
    pylab.plot(posa)
    pylab.plot(yy_corr)
    pylab.show()
#--------------------------------------------------------------------
# get difference of line cores to fixed value at 1/4 slit height
#--------------------------------------------------------------------
    ydiff = yy_corr - posa[(lasty-firsty)/4]

    rr = numpy.mean(ydiff)
    if (rr > 3):
        print(' deskew in reza.py LIB: ')
        print(' Warning ! Very high mean shift value of '+str(rr)+' pixels.')
        print(' Suggestion: check range of spectral lines/hairlines.')

#-------------------------------------------------------------------------
# shift all images to the default position at 1/4 slit height
#-------------------------------------------------------------------------
    #print(icorr.shape)
    #print(ydiff.shape)
    #stop
    for k in numpy.arange(firsty,lasty-3):
        print(k, k - firsty)
        gg = numpy.roll(icorr[k,:], int(-ydiff[k-firsty]))
        #print icorr.shape
        #print gg.shape
        #stop
        icorr[k,:] = gg

############################################################
# averages-out along the third axis of a fits file         #
############################################################
def average_axis3(kname):
    import glob, pyfits, numpy
    #anum = glob.glob('*'+kname+'*')
    #dc_img = numpy.zeros((dim[0], dim[1],), float)
    print(kname)
    a = pyfits.open(kname)
    d = a[0].data.copy()
    d = d * 1.0
    s = d.shape
    if (len(s) < 3)or(s[0] == 1):   # if it is a 2D image, and not 3D
        return
    print(s)
    dh = a[0].header.copy()
    ff = numpy.sum(d,0)/float(s[0])
    print(ff.shape)
    pyfits.writeto(kname+'a', ff, header=dh, clobber=True)
    return



class Song(object):
    def __init__(self, lyrics):
        self.lyrics = lyrics
    def sing_me_a_song(self):
        for line in self.lyrics:
            print(line)
happy_bday = Song(["Happy birthday to you",
                   "I don't want to get sued",
                   "So I'll stop right there"])
bulls_on_parade = Song(["They rally around tha family",
                        "With pockets full of shells"])

#happy_bday.sing_me_a_song()
#bulls_on_parade.sing_me_a_song()




def mean_confidence_interval(data, confidence):
    from scipy import stats
    a = 1.0*numpy.array(data)
    n = len(a)
    m, se = numpy.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def my_linear_func(x, a, b):
    return a * x + b

def ff(x, p):
    return my_linear_func(x, *p)

############################################################
# This program returns a chi-sq 2nd order fit to the data  #
# along with two curves for the upper and lower            #
# confidence levels                                        #
############################################################
def my_parabola_fit_confidence(xdata, ydata, conf_level, xout):
  from scipy import stats
  #
  def my_parabola_func(x, a, b, c):
    return a * x*x + b * x + c

  # Convert to percentile point of the normal distribution.
  # See: https://en.wikipedia.org/wiki/Standard_score
  pp = (1. + conf_level) / 2.
  # Convert to number of standard deviations.
  nstd = stats.norm.ppf(pp)
  #print(nstd)
  # Find best fit.
  popt, pcov = scipy.optimize.curve_fit(my_parabola_func, xdata, ydata)
  # Standard deviation errors on the parameters.
  perr = numpy.sqrt(np.diag(pcov))
  # Add nstd standard deviations to parameters
  #to obtain the confidence interval.
  popt_upp = popt + nstd * perr
  popt_dwn = popt - nstd * perr
  print('normal lsq fit: ', popt, perr)

  bestfit = my_parabola_func(xout, *popt)
  upp_curve = my_parabola_func(xout, *popt_upp)
  dwn_curve = my_parabola_func(xout, *popt_dwn)
  #print('fit: ', popt, 'error:', perr)
  return bestfit, upp_curve, dwn_curve

############################################################
# This program returns a chi-sq linaer fit to the data     #
# along with two curves for the upper and lower            #
# confidence levels                                        #
############################################################
def my_linear_fit_confidence(xdata, ydata, conf_level, xout):
  from scipy import stats
  #
  def my_linear_func(x, a, b):
    return a * x + b

  # Convert to percentile point of the normal distribution.
  # See: https://en.wikipedia.org/wiki/Standard_score
  pp = (1. + conf_level) / 2.
  # Convert to number of standard deviations.
  nstd = stats.norm.ppf(pp)
  #print(nstd)
  # Find best fit.
  popt, pcov = scipy.optimize.curve_fit(my_linear_func, xdata, ydata)
  # Standard deviation errors on the parameters.
  perr = numpy.sqrt(np.diag(pcov))
  # Add nstd standard deviations to parameters
  #to obtain the confidence interval.
  popt_upp = popt + nstd * perr
  popt_dwn = popt - nstd * perr
  print('normal lsq fit: ', popt, perr)

  bestfit = my_linear_func(xout, *popt)
  upp_curve = my_linear_func(xout, *popt_upp)
  dwn_curve = my_linear_func(xout, *popt_dwn)
  #print('fit: ', popt, 'error:', perr)
  return bestfit, upp_curve, dwn_curve

############################################################
# This program returns a robust linaer fit to the data     #
# and skips outlier data points.                           #
# can be extended to nonlinear fits                        #
"""
loss : str or callable, optional
    Determines the loss function. The following keyword values are allowed:

        * 'linear' (default) : ``rho(z) = z``. Gives a standard
          least-squares problem.
        * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
          approximation of l1 (absolute value) loss. Usually a good
          choice for robust least squares.
        * 'huber' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
          similarly to 'soft_l1'.
        * 'cauchy' : ``rho(z) = ln(1 + z)``. Severely weakens outliers
          influence, but may cause difficulties in optimization process.
        * 'arctan' : ``rho(z) = arctan(z)``. Limits a maximum loss on
          a single residual, has properties similar to 'cauchy'.

    If callable, it must take a 1-d ndarray ``z=f**2`` and return an
    array_like with shape (3, m) where row 0 contains function values,
    row 1 contains first derivatives and row 2 contains second
    derivatives. Method 'lm' supports only 'linear' loss.

f_scale : float, optional
    Value of soft margin between inlier and outlier residuals, default
    is 1.0. The loss function is evaluated as follows
    ``rho_(f**2) = C**2 * rho(f**2 / C**2)``, where ``C`` is `f_scale`,
    and ``rho`` is determined by `loss` parameter. This parameter has
    no effect with ``loss='linear'``, but for other `loss` values it is
    of crucial importance.
"""
############################################################
def my_robust_linear_fit(xdata, ydata, init_guess, xout):
  from scipy.optimize import least_squares
  #
  def cost_func(t, x, y):
    return t[0] * x + t[1] - y

  def real_func(x, t):
    return t[0] * x + t[1]

  res_lsq = least_squares(cost_func, init_guess, args=(xdata, ydata))
  #res_robust = least_squares(cost_func, init_guess, loss='soft_l1', f_scale=0.991, args=(xdata, ydata))
  res_robust = least_squares(cost_func, init_guess, loss='cauchy', f_scale=0.05, args=(xdata, ydata))
  #print(res_robust)
  lsq_params = res_lsq.x
  robust_params = res_robust.x
  print('standard lsq fit: ', lsq_params)
  print('robust lsq fit: ', robust_params)
  
  y_lsq = real_func(xout, lsq_params)
  y_robust = real_func(xout, robust_params)

  return y_lsq, y_robust, lsq_params, robust_params

############################################################
# This program returns a least_squares fit to the data     #
# and calculates parameter errors using the                #
# bootstrap method                                         #
# it is particularly helpful when the original fit         #
# does not return reasonable error of parameters.          #
############################################################
def fit_bootstrap(p0, datax, datay, function, yerr_systematic=0.0):
    from scipy import optimize
    from scipy.optimize import least_squares
    def real_func(x, t):
      return t[0] * x + t[1]
    errfunc = lambda p, x, y: function(x,p) - y

    # Fit first time
    pfit, perr = scipy.optimize.leastsq(errfunc, p0, args=(datax, datay), full_output=0)


    # Get the stdev of the residuals
    residuals = errfunc(pfit, datax, datay)
    sigma_res = numpy.std(residuals)

    sigma_err_total = numpy.sqrt(sigma_res**2 + yerr_systematic**2)
    # N random data sets are generated and fitted
    ps = []
    for i in range(300):

        randomDelta = numpy.random.normal(0., sigma_err_total, len(datay))
        randomdataY = datay + randomDelta

        randomfit, randomcov = \
            scipy.optimize.leastsq(errfunc, p0, args=(datax, randomdataY),full_output=0)
        ps.append(randomfit) 
        #p1 = p0 + numpy.random.normal(0.,5.5, len(p0))
        #randomfit = least_squares(errfunc, p0, loss='cauchy', f_scale=0.5, args=(datax, randomdataY))
        #randomfit = scipy.optimize.least_squares(errfunc, p0, args=(datax, randomdataY))
        #print(i, '  ', randomfit.x)
        #ps.append(randomfit.x) 

    ps = numpy.array(ps)
    #print(ps.shape)
    
    mean_pfit = numpy.mean(ps,0)

    # You can choose the confidence interval that you want for your
    # parameter estimates: 
    Nsigma = 1. # 1sigma gets approximately the same as methods above
                # 1sigma corresponds to 68.3% confidence interval
                # 2sigma corresponds to 95.44% confidence interval
    err_pfit = Nsigma * numpy.std(ps, 0) 

    pfit_bootstrap = mean_pfit
    perr_bootstrap = err_pfit

    return pfit_bootstrap, perr_bootstrap


############################################################
# This program returns the standard chi-squared of  data.  #
# chi = sum (y_data - y_fit)^2/sigma^2 / (N-1)             #
############################################################
def chi_sq(y_data,y_model, sigma=None):   
   if sigma==None:  
     chisq = numpy.sum((y_data-y_model)**2)  
   else:  
     chisq = numpy.sum( ((y_data-y_model)/sigma)**2 )  
   nu = len(y_data) - 1.
   return chisq/nu


############################################################
# This program returns the reduced chi-squared of  data.   #
# chi = sum (y_data - y_fit)^2/sigma^2                     #
# degree of freedome =                                     #
#  number of data points - 1 - number of free parameters   #
# reference: Bevington & Robinson, chapter 11              #
############################################################
def red_chi_sq(y_data,y_model,deg_free,sigma=None):  
   if sigma==None:  
     chisq = numpy.sum((y_data-y_model)**2)  
   else:  
     chisq = numpy.sum( ((y_data-y_model)/sigma)**2 )  
             
   nu = len(y_data) - 1. - deg_free  
        
   return chisq/nu 


############################################################
# read an ascii file as a whole at once                    #
############################################################
def read_ascii(infile):
    with open(infile) as f:
        array=[]
        array.append([float(x) for x in line.split()])
    return array

############################################################
# read an ascii file as line by line                       #
############################################################
def read_ascii_line(infile):
    with open(infile) as f:
        #next(f)  # to skip the first line
        array=[]
        for line in f:
            array.append([float(x) for x in line.split()])
    return array

############################################################
# read-in numeric ASCII table, similar to numpy.loadtxt    #
############################################################
def read_sample(infile, m, n):
    print(infile)
    array = zeros((m, n), float)
    with open(infile) as f:
        i = 0
        for line in f:
            new = [float(x) for x in line.split()]
            array[i, :] = new[0:n]
            i += 1
    array = numpy.asarray(array)
    return array

############################################################
# write an ascii file as a whole at once                   #
############################################################
def write_ascii(outfile, data):
    with open(outfile, 'w') as f:
      f.writelines(data)
    return

############################################################
# reverse a colormap                                       #
# https://stackoverflow.com/questions/3279560/invert-colormap-in-matplotlib
############################################################
def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In: 
    cmap, name 
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """        
    reverse = []
    k = []   

    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r


############################################################
# reverse a colormap                                       #
############################################################
def planck(lambd, temp):
    # lambd : lambda in nm
    # temp  : temperature in K
    # output : Planck function in [erg cm^-2 s^-1 nm^-1 sr^-1]
    # Original: Planck function in RH code of H. Uitenbroeck

    c_light = 2.99792458e+08
    h_planck = 6.626176e-34
    k_boltzmann = 1.380662e-23
    
    CM_TO_M      = 1.0e-02
    NM_TO_M      = 1.0e-09
    ERG_TO_JOULE = 1.0e-07

    lambda_m   = NM_TO_M * lambd # --- Convert lambda to [m] ---------------- 
    hc_kl      = (h_planck * c_light) / (k_boltzmann * lambda_m)
    twohnu3_c2 = (2.0 * h_planck * c_light) / lambda_m**3
    term = NM_TO_M * CM_TO_M**2 / ERG_TO_JOULE# ;; --- In [erg cm^-2 s^-1 nm^-1 sr^-1] -- 
    return term * twohnu3_c2 * (c_light/lambda_m**2) / (np.exp(hc_kl/temp) - 1.0)
'''
    to get the result in frequency version
    ;; --- In [J m^-2 s^-1 Hz^-1 sr^-1] --             --------------- ;

    return, twohnu3_c2 / (np.exp(hc_kl/temp) - 1.0)

'''
