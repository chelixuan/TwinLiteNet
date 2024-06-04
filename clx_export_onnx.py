import torch
import argparse
import onnx
from pathlib import Path

from model import TwinLite as net
from collections import OrderedDict

def run(
        weights,
        imgsz,
        batch_size,
        device,
        opset,
        onnx_path,
    ):

    model = net.TwinLiteNet()
    model = torch.nn.DataParallel(model)
    # model = model.cuda()

    state_dict = torch.load(weights)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:8] != 'module.':
            k = 'module.' + k
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    
    file = Path(weights)  # PyTorch weights
    # Input
    imgsz = [x for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection
    # Exports
    if not onnx_path:
        f = str(file)[:-4] + '_clx_opset-11.onnx'
    else:
        f = onnx_path
    
    torch.onnx.export(
        model.module,
        im, 
        f,
        opset_version = opset,
        input_names=["images"],
        output_names=["da", "ll"]
    )

    print('='*80)
    print(f'onnx export success : {f}')
    print('='*80)
    print()


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='pretrained/model_best.pth', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[360, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--opset', type=int, default=11, help='onnx opset version')
    parser.add_argument('--onnx_path', type=str, default=None, help='onnx save path')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

