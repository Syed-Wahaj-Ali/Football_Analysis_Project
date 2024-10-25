def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2] - bbox[0] # acc. bbox index x1 --> 0 , y1 --> 1, x2 --> 2, y2 --> 2

def measure_distance(pt1,pt2):
    return ((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**0.5

def measure_distance_cm(pt1,pt2):
    return (pt1[0]-pt2[0],pt1[1]-pt2[1])

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox 
    return int((x1+x2)/2), int(y2) 