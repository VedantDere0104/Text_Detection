import numpy as np
from skimage import io
import cv2
from torch.autograd import Variable
import torch
import time

from utils import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):

    show_time = True
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.to(device)

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    #print(polys)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = cvt2HeatmapImg(render_img)

    if show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
    #print(boxes , polys , ret_score_text)

    return boxes, polys, ret_score_text

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = CRAFT().to(device)
    network.load_state_dict(copyStateDict(torch.load('weights\craft_mlt_25k.pth' , map_location=device)))
    refine_network = RefineNet().to(device)
    refine_network.load_state_dict(copyStateDict(torch.load('weights\craft_refiner_CTW1500.pth' , map_location=device)))
    
    img = loadImage('test_\shop-name-board-500x500.jpg')
    #bboxes, polys, score_text = test_net(network, img  , args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_network)
    bboxes, polys, score_text = test_net(network, img, 0.7, 0.4, 0.4, device, False, refine_network)

    showResult(img , bboxes)