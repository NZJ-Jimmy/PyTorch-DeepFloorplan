import sys

import PIL
from tqdm import tqdm
# sys.path.append('./utils/')
from utils.rgb_ind_convertor import *
from utils.util import *
import cv2
from net import *
from data import *
import argparse 
import matplotlib.pyplot as plt

def BCHW2colormap(tensor,earlyexit=False):
    """
    Converts a BCHW tensor to a colormap representation.

    This function processes a tensor with shape (B, C, H, W) and converts it into a 
    colormap representation. If the `earlyexit` parameter is set to True, the function 
    returns the intermediate result before applying the argmax operation.

    Args:
        tensor (torch.Tensor): A 4D tensor with shape (B, C, H, W). If B > 1, only the 
                                first batch is processed.
        earlyexit (bool, optional): If True, returns the intermediate result after 
                                        squeezing and permuting the tensor. Defaults to False.

    Returns:
        numpy.ndarray: If `earlyexit` is False, returns a 2D array with shape (H, W) 
                        containing the argmax indices along the channel dimension. 
                        If `earlyexit` is True, returns a 3D array with shape (H, W, C) 
                        representing the intermediate colormap.
    """
    
    if tensor.size(0) != 1:
        tensor = tensor[0].unsqueeze(0)
    result = tensor.squeeze().permute(1,2,0).cpu().detach().numpy()
    if earlyexit:
        return result
    result = np.argmax(result,axis=2)
    return result

def initialize(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # data
    
    # benchmark file is a TSV file with the following format:
    # [orig]\t[wall]\t[door(close)]\t[room]\t[close_wall]
    # Each line is a test sample.
    sample = open(args.benchmark_path, 'r').read().splitlines()
    orig_paths = [p.split('\t')[0] for p in sample]
    origs = []  # original plan floor images
    images = [] # transformed images for model input
    trans = transforms.Compose([transforms.ToTensor()]) # transform for input
    
    for orig_path in orig_paths:
        orig_path = orig_path[1:]
        orig = cv2.imread(orig_path)
        orig = cv2.resize(orig,(512,512))
        image = trans(orig.astype(np.float32)/255.)
        image = image.unsqueeze(0).to(device)
        
        origs.append(orig)
        images.append(image)
        
    # model
    model = DFPmodel()
    model.load_state_dict(torch.load(args.loadmodel))
    model.to(device)
    
    return device,origs,images,model,orig_paths

def post_process(rm_ind,bd_ind):
    hard_c = (bd_ind>0).astype(np.uint8)
    # region from room prediction 
    rm_mask = np.zeros(rm_ind.shape)
    rm_mask[rm_ind>0] = 1
    # region from close wall line
    cw_mask = hard_c
    # regine close wall mask by filling the gap between bright line
    cw_mask = fill_break_line(cw_mask)

    fuse_mask = cw_mask + rm_mask
    fuse_mask[fuse_mask>=1] = 255

    # refine fuse mask by filling the hole
    fuse_mask = flood_fill(fuse_mask)
    fuse_mask = fuse_mask//255

    # one room one label
    new_rm_ind = refine_room_region(cw_mask,rm_ind)

    # ignore the background mislabeling
    new_rm_ind = fuse_mask*new_rm_ind

    return new_rm_ind


def main(args):

    device, origs, images, model, orig_paths = initialize(args)
    # run
    for image, orig_path in tqdm(zip(images, orig_paths)):
        with torch.no_grad():
            model.eval()
            logits_r,logits_cw = model(image)
            predroom = BCHW2colormap(logits_r) # The 9 classes of each pixel
            pred_cw = BCHW2colormap(logits_cw) # The 3 classes(space, door & window, wall)
        if args.postprocess:
            # postprocess
            predroom = post_process(predroom,pred_cw)
            
        rooms = ind2rgb(predroom,color_map=floorplan_fuse_map) # render the room prediction (index to RGB)
        door = (pred_cw == 1)   # door
        close_wall = (pred_cw != 0) # close wall (1: door, 2: wall)
        
        # Save room and cw
        orig_name = orig_path.split('/')[-1].removesuffix(".png").removesuffix(".jpg")
        room_output_path = f"out/room/{orig_name}_rooms.png"
        door_output_path = f"out/door/{orig_name}_close.png"
        cw_output_path = f"out/close_wall/{orig_name}_close_wall.png"
        
        plt.imsave(room_output_path, rooms.astype(np.uint8))
        plt.imsave(door_output_path, door , cmap = 'gray')
        plt.imsave(cw_output_path, close_wall , cmap = 'gray')
        
        print(orig_name)
        # plot
        # plt.subplot(1,3,1); plt.imshow(orig[:,:,::-1])
        # plt.subplot(1,3,2); plt.imshow(rgb)
        # plt.subplot(1,3,3); plt.imshow(predboundary)
        # plt.show()
    
    

# door(close), room(room), closewall(close + wall), im-result

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--loadmodel',type=str,default="log/store2/checkpoint.pt")
    p.add_argument('--postprocess',type=bool,default=False)
    # p.add_argument('--image_path',type=str,default="/home/jimmy/PyTorch-DeepFloorplan/dataset/newyork/test/9.jpg")
    p.add_argument('--benchmark_path', type=str, default="dataset/r3d_test.txt")
    args = p.parse_args()

    main(args)




