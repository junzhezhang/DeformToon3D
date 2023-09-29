from facexlib.alignment import init_alignment_model, landmark_98_to_7
from facexlib.visualization import visualize_alignment
import facexlib.utils.face_restoration_helper as face_restoration_helper
import torch

""""
NOTE script by Yushi on 2022/12/22
"""

class Demo():
    def __init__(self):
        self.FaceRestoreHelper = face_restoration_helper.FaceRestoreHelper(
            upscale_factor=1, face_size=256)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                 inplace=True)
        ])

    @torch.no_grad()
    def crop_one_img(FaceRestoreHelper, img, save_cropped_path=None):
        """input: cv2 img; 
        output aligned cv2 img
        """
        FaceRestoreHelper.clean_all()
        FaceRestoreHelper.read_image(img)
        # get face landmarks
        FaceRestoreHelper.get_face_landmarks_5() # ffhq by default
        return FaceRestoreHelper.align_warp_face(save_cropped_path)


    @torch.no_grad()
    def align_img(self, cv2_img: torch.Tensor):

        # align with ffhq landmarks, return the aligned img (in np array)
        assert cv2_img.shape[0] == 1 # cv2 img, bgr channel order, attention ! 
        cv2_img = cv2_img[0].cpu().numpy()
        aligned_cv2_img = crop_one_img(self.FaceRestoreHelper,
                                        cv2_img)[0]  # H W 3
        aligned_img_rgb = aligned_cv2_img[..., ::-1] / 255

        # transform to the tensor, [-1,1]
        aligned_rgb_Tensor = self.transform(aligned_img_rgb).to(
            self.device).unsqueeze(0)
        return aligned_cv2_img, aligned_rgb_Tensor