import os
import process
import config
import argparse
from DM.file_manage import RotatedNiiFileManager, LabelNiiFileManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A fast algorithm to calculate MPRI')
    parser.add_argument('mode', choices=['calc', 'seg', 'both'], help='''
    calc:                   Calculate and output the pons area, midbrain area, MCP
                            width, SCP width and MPRI.
    seg:                    Segment MCP, SCP and save to output image but not
                            calculate MPRI, etc. Note that --output OUTPUT must be
                            given in this mode.
    both:                   Equal to runing both calc mode and seg mode.
    ''')
    parser.add_argument('-i', '--image', required=True, help='The image path IMAGE to be segmented or calculated.')
    parser.add_argument('-l', '--label', required=True, help='''The lable path LABLE of correspounding image
                            from AccuBrain.''')
    parser.add_argument('-o', '--output', help='''The output path OUTPUT to save segmentation results.
                            It must be given if the mode is seg or both and does
                            not work if the mode is calc.''')
    parser.add_argument('-d', '--data', help='''The output path DATA to save calculation results. If
                            given, calculation results would be saved to DATA,
                            otherwise shown in the screen.''')
    args = parser.parse_args()
    assert os.path.isfile(args.label)
    assert os.path.isfile(args.image)
    image_nii = RotatedNiiFileManager(args.image)
    image_nii.load()
    image_nii.normalize()

    label_nii = LabelNiiFileManager(args.label)
    label_nii.load()
    show = args.mode != 'calc'

    quad_seg_point, mid_num, pons_area, midbrain_area, angle, rotate_quad_seg_point = process.first_stage.run(
        image_nii, label_nii
    )
    mcp_mean_width, mcp_show_info = process.second_stage.run(
        image_nii, label_nii, quad_seg_point, mid_num, show=show
    )
    scp_mean_width, scp_show_info = process.third_stage.run(
        image_nii, label_nii, rotate_quad_seg_point, mid_num, angle
    )
    if args.mode == 'calc':
        process.output.calc(pons_area, midbrain_area, mcp_mean_width, scp_mean_width, file_name=args.data)
    elif args.mode == 'seg':
        assert args.output
        process.output.save_to_sitk(
            args.output, image_nii.size, mid_num,
            quad_seg_point, mcp_show_info, scp_show_info
        )
        
    elif args.mode == 'both':
        assert args.output
        process.output.save_to_sitk(
            args.output, image_nii.size, mid_num,
            quad_seg_point, mcp_show_info, scp_show_info
        )
        process.output.calc(pons_area, midbrain_area, mcp_mean_width, scp_mean_width, file_name=args.data)
    else:
        parser.print_help()
