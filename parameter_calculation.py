import os
import numpy as np
from skimage import io, measure, transform
import pandas as pd
from math import pi, sqrt
from matplotlib import pyplot as plt
from numpy import ndarray


pix_mm=15.6


def get_att_points(img):
    contours = measure.find_contours(img)
    try:
        c_shape=np.shape(contours)
        contour = np.reshape(contours,(c_shape[0]*c_shape[1],2))
    except:
        contour=contours[0]
    xy=[]
    for i,row in enumerate(img):
        for j,pix in enumerate(row):
            if pix == 1:
                xy.append([j,i])
    xy=np.array(xy,ndmin=2)
    z = np.polyfit(contour[:,1],contour[:,0],3)
    f = np.poly1d(z)

    bottom_half=[]
    for point in contour:
        x=point[1]
        y=point[0]
        if y>f(x):
            bottom_half.append([x,y])
    bottom_half=np.array(bottom_half,ndmin=2)

    left = bottom_half[np.where(bottom_half[:,0]==min(bottom_half[:,0]))[0][0]]
    right = bottom_half[np.where(bottom_half[:,0]==max(bottom_half[:,0]))[0][0]]

    return left, right

def get_tcl_bounds(img):
    img_T = np.transpose(img)
    v,d,xy=[],[],[]
    for j,col in enumerate(img_T):
        v_found = False
        for i,pix in enumerate(col):
            if pix == True:
                xy.append([j,i])
            if v_found == False:
                if pix == True:
                    v_found = True
                    v.append([j,i])
            if v_found==True and pix == False:
                d.append([j,i])
                break
    v = np.array(v,ndmin=2)
    d = np.array(d,ndmin=2)
    xy = np.array(xy,ndmin=2)

    return d,v,xy


def get_d_boundary(img):
    contours = measure.find_contours(img)
    try:
        c_shape=np.shape(contours)
        contour = np.reshape(contours,(c_shape[0]*c_shape[1],2))
    except:
        contour=contours[0]
    d_boundary,v_boundary = [],[]
    for pt0 in contour:
        pt0=[int(pt0[0]),int(pt0[1])]
        v_pt,d_pt=[pt0[1],pt0[0]],[pt0[1],pt0[0]]
        skip=False
        for dpt in d_boundary:
            if pt0[1]==int(dpt[1]):
                skip=True
        if skip==True:
            continue
        for pt1 in contour:
            pt1=[int(pt1[0]),int(pt1[1])]
            if pt0[1]==pt1[1]:
                d_pt=[d_pt[0],max([d_pt[1],pt1[0]])]
                v_pt=[v_pt[0],min([v_pt[1],pt1[0]])]
        d_boundary.append(d_pt)
        v_boundary.append(v_pt)
    d_boundary=np.array(d_boundary,ndmin=2)
    v_boundary=np.array(v_boundary,ndmin=2)
    
    return d_boundary,v_boundary

def calc_t(d,v,left,right):
    l,r=left[0],right[0]
    length = r-l
    d = d[np.where(d[:,0]>l+length/3)]
    d = d[np.where(d[:,0]<r-length/3)]
    v = v[np.where(v[:,0]>l+length/3)]
    v = v[np.where(v[:,0]<r-length/3)]
    z = np.polyfit(d[:,0],d[:,1],2)
    f_d = np.poly1d(z)
    z = np.polyfit(v[:,0],v[:,1],2)
    f_v = np.poly1d(z)
    x = np.arange(int(l+length/3),int(r-length/3),1)
    ts=[]
    for pt in x:
        dy = f_d(pt)
        vy = f_v(pt)
        ts.append(dy-vy)
    ts=np.array(ts,ndmin=1)
    t = np.mean(ts)

    return t

def calc_t2(d,v,left,right):
    min_dists=[]
    l=right[0]-left[0]
    for p0 in d:
        if p0[0]<left[0]+l/3 or p0[0]>right[0]-l/3:
            continue
        dists=[]
        for p1 in v:
            if p0[0]-p1[0]>20:
                continue
            if p1[0]-p0[0]>20:
                continue
            dist = np.linalg.norm(p0-p1)

            dists.append(dist)
        min_dists.append(min(dists))
    min_dists=np.array(min_dists,ndmin=1)
    t = np.mean(min_dists)
    return t

def calc_CSA(img):
    contours = measure.find_contours(img)
    CSAs=[]
    for contour in contours:
        top = int(min(contour[:,0]))
        bottom = int(max(contour[:,0]))
        left = int(min(contour[:,1]))
        right = int(max(contour[:,1]))
        CSA=0
        for i in range(top-1,bottom+1):
            for j in range(left-1,right+1):
                if img[i,j]==True:
                    CSA+=1
        CSA=CSA
        CSAs.append(CSA)
    CSA=max(CSAs)    
    return(CSA)

def calc_P(img):
    P = measure.perimeter_crofton(img)
    return P

def calc_C(CSA,P):
    C = 4*pi*CSA/P**2
    return C

def calc_CAW(L0,R0,d):

    Rx,Ry = R0[0],R0[1]
    Lx,Ly = L0[0],L0[1]

    if R0[1]==L0[1]:
        R1 = [Rx,Ry+d]
        R1 = np.array(R1,ndmin=1)
        L1 = [Lx,Ly+d]
        L1 = np.array(L1,ndmin=1)
    
    else:
        m = -(R0[0]-L0[0])/(R0[1]-L0[1])
        r = sqrt(1+m**2)
        if m<0:
            r=-r        
        R1 = [Rx+d/r,Ry+d*m/r]
        R1 = np.array(R1,ndmin=1)
        L1 = [Lx+d/r,Ly+d*m/r]
        L1 = np.array(L1,ndmin=1)

    CAW = np.linalg.norm(R1-L1)

    return L1,R1,CAW


def calc_CAH(img,left,right,d_boundary):
    dists=[]
    p0=right 
    p1=left
    x=right[0]-left[0] 

    for pd in d_boundary:
        if pd[0]<p1[0]+x/5 or pd[0]>right[0]-x/5:
            continue
        d=-np.cross(p0-p1,pd-p1)/np.linalg.norm(p0-p1)
        dists.append(d)   
    CAH = max(dists)

    return(CAH)

def calc_CAA(img, left, right):
    x=right[0]-left[0]
    m=(right[1]-left[1])/(right[0]-left[0])
    b=right[1]-m*right[0]
    CAW_pts=[]
    CAA=0
    for x in range(int(left[0]+x/5),int(right[0]-x/5)):
        y=m*x+b 
        CAW_pts.append([x,y])
        count=0
        for row in range(int(y),0,-1):
            if img[row,int(x)]!=0:
                break
            count+=1
            if row == 1:
                count=0
        CAA+=count

    return CAA

def calc_mdn_tcl_dist(img,d):
   row, col = img.shape

   xVal = 0
   yVal = 0
   n = 0.0
   xy=[]

   for i in range(0,row):
      for j in range(0,col):
         if (img[i,j] == True):
            xy.append([j,i])
            xVal += j
            yVal += i
            n += 1.0

   xy=np.array(xy,ndmin=2)
   xVal /= n
   yVal /= n
   centroid = np.array([xVal,yVal],ndmin=1)


   dists=[]
   for pt in d:
      dists.append(np.linalg.norm(pt-centroid))
   d = min(dists)

   return d,xy 

def gs_bin(img):
    thresh=float(max(img.flatten()))+float(min(img.flatten()))
    thresh=thresh/5
    for i,row in enumerate(img):
        for j,pix in enumerate(row):
            if pix>thresh:
                img[i,j]=True
            if pix<thresh:
                img[i,j]=False
    return img




# Set the path to the folder containing the binary labels
data_type = 'gt'
labels_folder = ".\Labels_gs"
img_folder="./mdn"
mdn_folder='./mdn2'
tcl_folder='./tcl'
file_ends = [['.jpg','.jpg'],['_mdn.jpg','_tcl.jpg']]

# Get a list of all files in the folder
files = os.listdir(img_folder)

# Initialize an empty list to store the binary labels
labels = []

CSAs,Ps,Cs,CAWs,CAHs,CAAs,ts,distances,sub_ids=[],[],[],[],[],[],[],[],[]
# Loop over all the files in the folder
for file in files:

    sub_id = file[0:9]

    if sub_id in sub_ids:
        continue
    # Load the binary label image using the scikit-image io.imread function
    mdn_gt_label = io.imread(os.path.join(labels_folder, sub_id + '_mdn.jpg'),as_gray=True)
    tcl_gt_label = io.imread(os.path.join(labels_folder, sub_id + '_tcl.jpg'),as_gray=True)
    if data_type == 'pred':
        mdn_label = io.imread(os.path.join(mdn_folder, sub_id + '.jpg'),as_gray=True)
        tcl_label = io.imread(os.path.join(tcl_folder, sub_id + '.jpg'),as_gray=True)
        mdn_label = transform.resize(mdn_label,np.shape(mdn_gt_label))
        tcl_label = transform.resize(tcl_label,np.shape(tcl_gt_label))


    if data_type == 'gt':
        mdn_label = io.imread(os.path.join(labels_folder, sub_id + '_mdn.jpg'),as_gray=True)
        tcl_label = io.imread(os.path.join(labels_folder, sub_id + '_tcl.jpg'),as_gray=True)


    mdn_label=gs_bin(mdn_label)
    tcl_label=gs_bin(tcl_label)
    #plt.imshow(tcl_label)
    #plt.show()
    print(sub_id)

    pix_mm=15.6
    
    d_boundary,v_boundary,tcl_xy = get_tcl_bounds(tcl_label)
    left = tcl_xy[np.where(tcl_xy[:,0]==min(tcl_xy[:,0]))][0]
    right = tcl_xy[np.where(tcl_xy[:,0]==max(tcl_xy[:,0]))][0]


    #CSA = calc_CSA(mdn_label)
    

    P = calc_P(mdn_label)
    P=P/15.6
    Ps.append(P)

    dist,mdn_xy = calc_mdn_tcl_dist(mdn_label,d_boundary)
    dist = dist/pix_mm
    distances.append(dist)

    CSA = np.shape(mdn_xy)[0]
    CSA=CSA/pix_mm**2
    CSAs.append(CSA)

    t = calc_t2(d_boundary,v_boundary,left,right)
    t = t/pix_mm
    ts.append(t)

    L,R,CAW = calc_CAW(left,right,t*pix_mm)
    CAW = CAW/pix_mm
    CAWs.append(CAW)

    CAH = calc_CAH(tcl_label,L,R,d_boundary)
    CAH=CAH/pix_mm
    CAHs.append(CAH)

    CAA = calc_CAA(tcl_label,L,R)
    CAA=CAA/pix_mm**2
    CAAs.append(CAA)

    C = calc_C(CSA,P)
    Cs.append(C)
    
    sub_ids.append(sub_id)


d={'sub_id':sub_ids,'mdn_CSA':CSAs,'mdn_P':Ps,'mdn_C':Cs,'CAW':CAWs,'CAH':CAHs,'CAA':CAAs,'t':ts,'dist':distances}
results_df = pd.DataFrame(d,columns=['sub_id','mdn_CSA','mdn_P','mdn_C','CAW','CAH','CAA','t','dist'])
print(results_df)
try:
    with pd.ExcelWriter('Morph_params.xlsx',mode='a',if_sheet_exists='replace') as writer:
        results_df.to_excel(writer,sheet_name=data_type)
except:
    results_df.to_excel('Morph_params.xlsx',sheet_name=data_type)
