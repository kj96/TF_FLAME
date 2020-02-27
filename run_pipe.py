import fit_2D_landmarks as f2d
import build_texture_from_image as bti


target_dir  =   '/data/me/jeff/'
target_name =   'jeff1'
target_file =   target_dir + target_name

## Fit 2D landmarks
# python fit_2D_landmarks.py
#           --tf_model_fname './models/female_model'
#           --template_fname './data/template.ply'
#           --flame_lmk_path './data/flame_static_embedding.pkl'
#           --texture_mapping './data/texture_data.npy'
#           --target_img_path './data/imgHQ00088.jpeg'
#           --target_lmk_path './data/imgHQ00088_lmks.npy'
#           --out_path './results'


tf_model    =   './models/generic_model'
mesh_temp   =   './data/template.ply'
lmk_path    =   './data/flame_static_embedding.pkl'
texr_map    =   './data/texture_data.npy'
out_path    =   './results/'

taret_img   =   target_file + '.png'
target_lmk  =   target_file + '.npy'


f2d.run_2d_lmk_fitting(tf_model, mesh_temp, lmk_path, texr_map, taret_img, target_lmk, out_path)
print("[SUCCESS] fit FLAME to 2D landmarks")


## Create textured mesh
# python build_texture_from_image.py
#           --source_img './data/imgHQ00088.jpeg'
#           --target_mesh './results/imgHQ00088.obj'
#           --target_scale './results/imgHQ00088_scale.npy'
#           --texture_mapping './data/texture_data.npy'
#           --out_path './results'

source_img = taret_img
target_mesh = out_path + target_name + '.obj'
target_scale = out_path + target_name + '_scale.obj'
texture_mapping = './data/texture_data.npy'
out_path = out_path

# bti.build_texture_from_image(source_img, target_mesh, target_scale, texture_mapping, out_path)
# print("[SUCCESS] Created textured mesh")



## Fit 3D Landmarks
