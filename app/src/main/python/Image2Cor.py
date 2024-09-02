import cv2  # opencv-python版本4.10.0.84
import numpy as np  # numpy版本2.0.1
# from pyexiv2 import Image   # pyexiv2版本2.12.0
import xml.etree.ElementTree as ET
import math
import json

def Image2Cor(json1):
# def image2Cor(loc0, loc_ganta, loc_mov1, loc_ref, loc_mov2,
#               degree_mov1, degree_ref, degree_mov2, byteArray0, byteArray1, byteArray2,
#               box_obj_mov1, box_obj_ref, box_obj_mov2):
    '''
    输入参数部分
    '''

    data = json.loads(json1)
    loc0 = [data["startLoc"]["lat"], data["startLoc"]["lon"], data["startLoc"]["alt"]]
    loc_ganta = [data["towerLoc"]["lat"], data["towerLoc"]["lon"], data["towerLoc"]["alt"]]
    loc_mov1 = [data["photoData"][0]["location"]["lat"], data["photoData"][0]["location"]["lon"], data["photoData"][0]["location"]["alt"]]
    loc_ref = [data["photoData"][1]["location"]["lat"], data["photoData"][1]["location"]["lon"], data["photoData"][1]["location"]["alt"]]
    loc_mov2 = [data["photoData"][2]["location"]["lat"], data["photoData"][2]["location"]["lon"], data["photoData"][2]["location"]["alt"]]
    degree_mov1 = [data["photoData"][0]["roll"], data["photoData"][0]["yaw"], data["photoData"][0]["pitch"]]
    degree_ref = [data["photoData"][1]["roll"], data["photoData"][1]["yaw"], data["photoData"][1]["pitch"]]
    degree_mov2 = [data["photoData"][2]["roll"], data["photoData"][2]["yaw"], data["photoData"][2]["pitch"]]
    byteArray0 = data["photoData"][0]["data"]
    byteArray1 = data["photoData"][1]["data"]
    byteArray2 = data["photoData"][2]["data"]
    box_obj_mov1 = data["photoData"][0]["objs"]
    box_obj_ref = data["photoData"][1]["objs"]
    box_obj_mov2 = data["photoData"][2]["objs"]

    # 输入参数——原点坐标地址，杆塔坐标
    # loc0 = [30.3067, 114.4250, 38.7490]
    loc_ganta = [30.3067, 114.4250, 38.7490]
    xyz_ganta = WGS84_to_ENU(loc0, loc_ganta)
    xyz_ganta = [ 6.6,  7, -7]
    # 输入参数——连续拍摄的三张图像path_mov1, path_ref, path_mov2
    path_mov1 = "/storage/emulated/0/etower/test0.jpg"
    path_ref = "/storage/emulated/0/etower/test1.jpg"
    path_mov2 = "/storage/emulated/0/etower/test2.jpg"

    # img_mov1_ori = cv2.imread(path_mov1)
    # img_ref_ori = cv2.imread(path_ref)
    # img_mov2_ori = cv2.imread(path_mov2)
    nparr = np.frombuffer(byteArray0, np.uint8)
    img_mov1_ori = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    nparr = np.frombuffer(byteArray1, np.uint8)
    img_ref_ori = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    nparr = np.frombuffer(byteArray2, np.uint8)
    img_mov2_ori = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 输入参数——边缘两张图像对应的经纬高loc_mov1, loc_mov2
    # 输入参数——边缘两张图像对应的云台角度degree_mov1, degree_mov2
    # 输入参数——边缘两张图像对应的无人机角度degree_flight_mov1, degree_flight_mov2
    # loc_mov1 = [30.3067055, 114.4250684, 38.777];
    # loc_ref = [30.3067068, 114.4250727, 38.771];
    # loc_mov2 = [30.3067065, 114.4250782, 38.755];
    degree_mov1 = [0, -2.3, -54.4];
    degree_ref = [0, -5.1, -54.4];
    degree_mov2 = [0, -11, -54.4];

    # 输入参数——图像识别结果list_obj_mov1, list_obj_ref, list_obj_mov2
    # box_obj_mov1 = Box_Obj_Statistics(path_mov1[:-3] + "xml")
    # box_obj_ref = Box_Obj_Statistics(path_ref[:-3] + "xml")
    # box_obj_mov2 = Box_Obj_Statistics(path_mov2[:-3] + "xml")

    box_obj_mov1 = ConvertBox(box_obj_mov1)
    box_obj_ref = ConvertBox(box_obj_ref)
    box_obj_mov2 = ConvertBox(box_obj_mov2)

    '''
    可调整参数部分
    '''
    # 图像预处理参数
    # long_cor = 1500     # 长边剪裁位置
    # short_cor = 1000    # 短边剪裁位置
    # step = 1000         # 剪裁大小
    long_cor = 500     # 长边剪裁位置
    short_cor = 1000    # 短边剪裁位置
    step = 2000         # 剪裁大小
    scale1 = 0.5       # 图像缩放比例

    # 光流匹配流程参数
    speed_OpticalFlow = 2   # 光流速度，0-4依次减慢
    threshold_cut = 30     # 剪裁尺寸下光流物料匹配的重合度像素阈值，大于该阈值的认为没有光流结果与之匹配
    scale_jyz = 0.25       # 取识别框的中心区域处相对于原始识别框scale2大小的图像进行绝缘子光流匹配
    scale_blq = 1       # 取识别框的中心区域处相对于原始识别框scale2大小的图像进行绝缘子光流匹配
    scale_rdq = 1       # 取识别框的中心区域处相对于原始识别框scale2大小的图像进行绝缘子光流匹配
    scale_hd = 1        # 取识别框的中心区域处相对于原始识别框scale2大小的图像进行绝缘子光流匹配
    scale_byq = 0.25    # 取识别框的中心区域处相对于原始识别框scale2大小的图像进行绝缘子光流匹配
    visual_option = 0   # 光流匹配可视化开关
    threshold2 = 20     # 原始尺寸下去重光流匹配结果的像素阈值，当有多个光流匹配到同一物料时应用，与threshold1同理

    # 空间测距流程参数
    x_resolution = 4056
    y_resolution= 3040
    d = 2903
    '''
    图像预处理
    '''
    img_mov1_cut = Image_preprocessing(img_mov1_ori, long_cor, short_cor, step, scale1)
    img_ref_cut = Image_preprocessing(img_ref_ori, long_cor, short_cor, step, scale1)
    img_mov2_cut = Image_preprocessing(img_mov2_ori, long_cor, short_cor, step, scale1)

    # 计算两幅图间的光流
    flow1 = OpticalFlow(img_ref_cut, img_mov1_cut, speed_OpticalFlow)
    flow2 = OpticalFlow(img_ref_cut, img_mov2_cut, speed_OpticalFlow)

    # 将原始尺寸识别框转换为剪裁后尺寸
    box_obj_mov1_cut = Translate_scale(box_obj_mov1, long_cor, short_cor, scale1)
    box_obj_ref_cut = Translate_scale(box_obj_ref, long_cor, short_cor, scale1)
    box_obj_mov2_cut = Translate_scale(box_obj_mov2, long_cor, short_cor, scale1)

    # 计算三幅图对应无人机的坐标
    xyz_mov1 = WGS84_to_ENU(loc0, loc_mov1)
    xyz_ref = WGS84_to_ENU(loc0, loc_ref)
    xyz_mov2 = WGS84_to_ENU(loc0, loc_mov2)
    '''
    对绝缘子物料进行匹配
    '''
    # 绝缘子物料匹配
    jyz_matched1, jyz_dist_list1, jyz_index_list1 = Objs_match(
        box_obj_ref_cut, box_obj_mov1_cut, img_ref_cut, img_mov1_cut, flow1, 'jueyuanzi', threshold_cut, scale_jyz, visual_option
    )
    jyz_matched2, jyz_dist_list2, jyz_index_list2 = Objs_match(
        box_obj_ref_cut, box_obj_mov2_cut, img_ref_cut, img_mov2_cut, flow2, 'jueyuanzi', threshold_cut, scale_jyz, visual_option
    )

    # 将绝缘子物料匹配结果还原到原始尺寸
    jyz_matched_ori1 = Restore_match(jyz_matched1, long_cor, short_cor, scale1)
    jyz_matched_ori2 = Restore_match(jyz_matched2, long_cor, short_cor, scale1)

    # 对绝缘子物料匹配结果进行去重
    jyz_matched_derepeat1 = Matched_Derepeat(
        jyz_matched_ori1, box_obj_ref_cut, box_obj_mov1_cut, jyz_index_list1, jyz_dist_list1, 'jueyuanzi', long_cor, short_cor, scale1, threshold2
    )
    jyz_matched_derepeat2 = Matched_Derepeat(
        jyz_matched_ori2, box_obj_ref_cut, box_obj_mov2_cut, jyz_index_list2, jyz_dist_list2, 'jueyuanzi', long_cor, short_cor, scale1, threshold2
    )

    '''
    对绝缘子空间测距
    '''
    # 对mov1和mov2进行三角测量
    jyz_3d_pts = Calculate_3d_pts(jyz_matched_derepeat1, jyz_matched_derepeat2, degree_mov1, degree_mov2, xyz_mov1, xyz_mov2, x_resolution, y_resolution, d)

    '''
    对避雷器物料进行匹配
    '''
    # 当识别到避雷器时
    if 'bileiqi' in box_obj_ref_cut and 'bileiqi' in box_obj_mov1_cut and 'bileiqi' in box_obj_mov2_cut:
        # 进行避雷器物料匹配
        blq_matched1, blq_dist_list1, blq_index_list1 = Objs_match(
            box_obj_ref_cut, box_obj_mov1_cut, img_ref_cut, img_mov1_cut, flow1, 'bileiqi', threshold_cut, scale_blq, visual_option
        )
        blq_matched2, blq_dist_list2, blq_index_list2 = Objs_match(
            box_obj_ref_cut, box_obj_mov2_cut, img_ref_cut, img_mov2_cut, flow2, 'bileiqi', threshold_cut, scale_blq, visual_option
        )
        # 将避雷器物料匹配结果还原到原始尺寸
        blq_matched_ori1 = Restore_match(blq_matched1, long_cor, short_cor, scale1)
        blq_matched_ori2 = Restore_match(blq_matched2, long_cor, short_cor, scale1)

        # 对熔断器物料匹配结果进行去重
        blq_matched_derepeat1 = Matched_Derepeat(
            blq_matched_ori1, box_obj_ref_cut, box_obj_mov1_cut, blq_index_list1, blq_dist_list1, 'bileiqi', long_cor, short_cor, scale1, threshold2
        )
        blq_matched_derepeat2 = Matched_Derepeat(
            blq_matched_ori2, box_obj_ref_cut, box_obj_mov2_cut, blq_index_list2, blq_dist_list2, 'bileiqi', long_cor, short_cor, scale1, threshold2
        )

        '''
        对避雷器空间测距
        '''
        # 对mov1和mov2进行三角测量
        blq_3d_pts = Calculate_3d_pts(blq_matched_derepeat1, blq_matched_derepeat2, degree_mov1, degree_mov2, xyz_mov1, xyz_mov2, x_resolution, y_resolution, d)
        num_blq = blq_3d_pts.shape[0]
    else:
        num_blq = 0

    '''
    对熔断器物料进行匹配
    '''
    # 当识别到熔断器时
    if 'rongduanqi' in box_obj_ref_cut and 'rongduanqi' in box_obj_mov1_cut and 'rongduanqi' in box_obj_mov2_cut:
        # 进行熔断器物料匹配
        rdq_matched1, rdq_dist_list1, rdq_index_list1 = Objs_match(
            box_obj_ref_cut, box_obj_mov1_cut, img_ref_cut, img_mov1_cut, flow1, 'rongduanqi', threshold_cut, scale_rdq, visual_option
        )
        rdq_matched2, rdq_dist_list2, rdq_index_list2 = Objs_match(
            box_obj_ref_cut, box_obj_mov2_cut, img_ref_cut, img_mov2_cut, flow2, 'rongduanqi', threshold_cut, scale_rdq, visual_option
        )
        # 将熔断器物料匹配结果还原到原始尺寸
        rdq_matched_ori1 = Restore_match(rdq_matched1, long_cor, short_cor, scale1)
        rdq_matched_ori2 = Restore_match(rdq_matched2, long_cor, short_cor, scale1)

        # 对熔断器物料匹配结果进行去重
        rdq_matched_derepeat1 = Matched_Derepeat(
            rdq_matched_ori1, box_obj_ref_cut, box_obj_mov1_cut, rdq_index_list1, rdq_dist_list1, 'rongduanqi', long_cor, short_cor, scale1, threshold2
        )
        rdq_matched_derepeat2 = Matched_Derepeat(
            rdq_matched_ori2, box_obj_ref_cut, box_obj_mov2_cut, rdq_index_list2, rdq_dist_list2, 'rongduanqi', long_cor, short_cor, scale1, threshold2
        )
        '''
        对熔断器空间测距
        '''
        # 对mov1和mov2进行三角测量
        rdq_3d_pts = Calculate_3d_pts(rdq_matched_derepeat1, rdq_matched_derepeat2, degree_mov1, degree_mov2, xyz_mov1, xyz_mov2, x_resolution, y_resolution, d)
        num_rdq = rdq_3d_pts.shape[0]
    else:
        num_rdq = 0

    '''
    对横担物料进行匹配
    '''
    # 当识别到横担时
    if 'hengdan' in box_obj_ref_cut and 'hengdan' in box_obj_mov1_cut and 'hengdan' in box_obj_mov2_cut:
        # 进行横担物料匹配
        hd_matched1, hd_dist_list1, hd_index_list1 = Objs_match(
            box_obj_ref_cut, box_obj_mov1_cut, img_ref_cut, img_mov1_cut, flow1, 'hengdan', threshold_cut, scale_hd, visual_option
        )
        hd_matched2, hd_dist_list2, hd_index_list2 = Objs_match(
            box_obj_ref_cut, box_obj_mov2_cut, img_ref_cut, img_mov2_cut, flow2, 'hengdan', threshold_cut, scale_hd, visual_option
        )

        # 将横担物料匹配结果还原到原始尺寸
        hd_matched_ori1 = Restore_match(hd_matched1, long_cor, short_cor, scale1)
        hd_matched_ori2 = Restore_match(hd_matched2, long_cor, short_cor, scale1)

        # 对横担物料匹配结果进行去重

        hd_matched_derepeat1 = Matched_Derepeat(
            hd_matched_ori1, box_obj_ref_cut, box_obj_mov1_cut, hd_index_list1, hd_dist_list1, 'hengdan', long_cor, short_cor, scale1, threshold2
        )

        hd_matched_derepeat2 = Matched_Derepeat(
            hd_matched_ori2, box_obj_ref_cut, box_obj_mov2_cut, hd_index_list2, hd_dist_list2, 'hengdan', long_cor, short_cor, scale1, threshold2
        )

        '''
        对横担空间测距
        '''
        # 对mov1和mov2进行三角测量
        hd_3d_pts = Calculate_3d_pts(hd_matched_derepeat1, hd_matched_derepeat2, degree_mov1, degree_mov2, xyz_mov1, xyz_mov2, x_resolution, y_resolution, d)
        num_hd = hd_3d_pts.shape[0]

    else:
        num_hd = 0

    '''
    对变压器进行匹配
    '''
    if 'bianyaqi' in box_obj_ref_cut and 'bianyaqi' in box_obj_mov1_cut and 'bianyaqi' in box_obj_mov2_cut:
        byq_matched1, byq_dist_list1, byq_index_list1 = Objs_match(
            box_obj_ref_cut, box_obj_mov1_cut, img_ref_cut, img_mov1_cut, flow1, 'bianyaqi', threshold_cut, scale_byq, visual_option
        )
        byq_matched2, byq_dist_list2, byq_index_list2 = Objs_match(
            box_obj_ref_cut, box_obj_mov2_cut, img_ref_cut, img_mov2_cut, flow2, 'bianyaqi', threshold_cut, scale_byq, visual_option
        )

        # 将变压器物料匹配结果还原到原始尺寸
        byq_matched_ori1 = Restore_match(byq_matched1, long_cor, short_cor, scale1)
        byq_matched_ori2 = Restore_match(byq_matched2, long_cor, short_cor, scale1)

        # 对变压器物料匹配结果进行去重
        byq_matched_derepeat1 = Matched_Derepeat(
            byq_matched_ori1, box_obj_ref_cut, box_obj_mov1_cut, byq_index_list1, byq_dist_list1, 'bianyaqi', long_cor, short_cor, scale1, threshold2
        )

        byq_matched_derepeat2 = Matched_Derepeat(
            byq_matched_ori2, box_obj_ref_cut, box_obj_mov2_cut, byq_index_list2, byq_dist_list2, 'bianyaqi', long_cor, short_cor, scale1, threshold2
        )

        '''
        对变压器空间测距
        '''
        # 对mov1和mov2进行三角测量
        byq_3d_pts = Calculate_3d_pts(byq_matched_derepeat1, byq_matched_derepeat2, degree_mov1, degree_mov2, xyz_mov1, xyz_mov2, x_resolution, y_resolution, d)
        num_byq = byq_3d_pts.shape[0]
    else:
        num_byq = 0

    '''
    对杆塔本体进行匹配
    '''
    ganta_bottom = []
    ganta_top = []
    if 'ganta' in box_obj_mov1:
        # 提取box包含图像中心的杆塔本体
        box_ganta_mov1 = box_obj_mov1['ganta']
        box_ganta_mov1_a = [row for row in box_ganta_mov1
                            if row[0] <= (x_resolution / 2) and row[1] <= (y_resolution / 2)
                            and row[2] >= (x_resolution / 2) and row[3] >= (y_resolution / 2)]
        box_ganta_mov1_a = np.array(box_ganta_mov1_a)
        # 提取box中离中心更近的框

        nearest_index = np.argmin(abs(box_ganta_mov1_a[:, 4] - x_resolution / 2))
        box_ganta_mov1_s = box_ganta_mov1_a[nearest_index]
        # 计算杆塔的底
        vec_bottom = Pixel_to_angle((box_ganta_mov1_s[2] - box_ganta_mov1_s[0]) / 2 + x_resolution / 2, box_ganta_mov1_s[3], x_resolution, y_resolution, d)
        vec_world_bottom = Camera_vec_to_world_vec(vec_bottom, degree_mov1[0], degree_mov1[1], degree_mov1[2])
        ganta_bottom_mov1 = Calculate_point_in_L_and_norm([0, 0, 1], xyz_ganta, vec_world_bottom, xyz_mov1)
        ganta_bottom.append(ganta_bottom_mov1)
        # 计算杆塔的高

        vec_top = Pixel_to_angle((box_ganta_mov1_s[2] - box_ganta_mov1_s[0]) / 2 + x_resolution / 2, box_ganta_mov1_s[1], x_resolution, y_resolution, d)
        vec_world_top = Camera_vec_to_world_vec(vec_top, degree_mov1[0], degree_mov1[1], degree_mov1[2])

        ganta_top_mov1 = Calculate_point_in_L_and_norm([0, 0, 1], xyz_ganta, vec_world_top, xyz_mov1)
        ganta_top.append(ganta_top_mov1)

    if 'ganta' in box_obj_ref:
        box_ganta_ref = box_obj_ref['ganta']
        box_ganta_ref_a = [row for row in box_ganta_ref
                           if row[0] <= (x_resolution / 2) and row[1] <= (y_resolution / 2)
                           and row[2] >= (x_resolution / 2) and row[3] >= (y_resolution / 2)]
        box_ganta_ref_a = np.array(box_ganta_ref_a)
        nearest_index = np.argmin(abs(box_ganta_ref_a[:, 4] - x_resolution / 2))
        box_ganta_ref_s = box_ganta_mov1_a[nearest_index]
        # 计算杆塔的底
        vec_bottom = Pixel_to_angle((box_ganta_ref_s[2] - box_ganta_ref_s[0]) / 2 + x_resolution / 2, box_ganta_ref_s[3], x_resolution, y_resolution, d)
        vec_world_bottom = Camera_vec_to_world_vec(vec_bottom, degree_ref[0], degree_ref[1], degree_ref[2])
        ganta_bottom_ref = Calculate_point_in_L_and_norm([0, 0, 1], xyz_ganta, vec_world_bottom, xyz_ref)
        ganta_bottom.append(ganta_bottom_ref)
        # 计算杆塔的高
        vec_top = Pixel_to_angle((box_ganta_ref_s[2] - box_ganta_ref_s[0]) / 2 + x_resolution / 2, box_ganta_ref_s[1], x_resolution, y_resolution, d)
        vec_world_top = Camera_vec_to_world_vec(vec_top, degree_ref[0], degree_ref[1], degree_ref[2])
        ganta_top_ref = Calculate_point_in_L_and_norm([0, 0, 1], xyz_ganta, vec_world_top, xyz_ref)
        ganta_top.append(ganta_top_ref)

    if 'ganta' in box_obj_mov2:
        box_ganta_mov2 = box_obj_mov2['ganta']
        box_ganta_mov2_a = [row for row in box_ganta_mov2
                            if row[0] <= (x_resolution / 2) and row[1] <= (y_resolution / 2)
                            and row[2] >= (x_resolution / 2) and row[3] >= (y_resolution / 2)]
        box_ganta_mov2_a = np.array(box_ganta_mov2_a)
        nearest_index = np.argmin(abs(box_ganta_mov2_a[:, 4] - x_resolution / 2))
        box_ganta_mov2_s = box_ganta_mov1_a[nearest_index]
        # 计算杆塔的底
        vec_bottom = Pixel_to_angle((box_ganta_mov2_s[2] - box_ganta_mov2_s[0]) / 2 + x_resolution / 2, box_ganta_mov2_s[3], x_resolution, y_resolution, d)
        vec_world_bottom = Camera_vec_to_world_vec(vec_bottom, degree_mov2[0], degree_mov2[1], degree_mov2[2])
        ganta_bottom_mov2 = Calculate_point_in_L_and_norm([0, 0, 1], xyz_ganta, vec_world_bottom, xyz_mov2)
        ganta_bottom.append(ganta_bottom_mov2)
        # 计算杆塔的高
        vec_top = Pixel_to_angle((box_ganta_mov2_s[2] - box_ganta_mov2_s[0]) / 2 + x_resolution / 2, box_ganta_mov2_s[1], x_resolution, y_resolution, d)
        vec_world_top = Camera_vec_to_world_vec(vec_top, degree_mov2[0], degree_mov2[1], degree_mov2[2])
        ganta_top_mov2 = Calculate_point_in_L_and_norm([0, 0, 1], xyz_ganta, vec_world_top, xyz_mov2)
        ganta_top.append(ganta_top_mov2)

    '''
    记录物料空间位置
    '''
    obj_3d_pts = {}
    obj_3d_pts['jueyuanzi'] = jyz_3d_pts
    ganta_top = np.array(ganta_top)
    ganta_bottom = np.array(ganta_bottom)

    ganta_h = ganta_top[:, 2].reshape(-1, 1) - ganta_bottom[:, 2].reshape(-1, 1)
    # key = ganta时每一行为[杆塔顶端，杆塔底部，杆塔高度]
    obj_3d_pts['ganta'] = np.hstack((ganta_top[:, 2].reshape(-1, 1), ganta_bottom[:, 2].reshape(-1, 1), ganta_h))

    if num_blq != 0:
        obj_3d_pts['bileiqi'] = blq_3d_pts
    if num_rdq != 0:
        obj_3d_pts['rongduanqi'] = rdq_3d_pts
    if num_hd != 0:
        obj_3d_pts['hengdan'] = hd_3d_pts
    if num_byq != 0:
        obj_3d_pts['bianyaqi'] = byq_3d_pts

    data_converted = {key: value.tolist() for key, value in obj_3d_pts.items()}

    json_string = json.dumps(data_converted)
    return json_string

def ConvertBox(Boxes):
    Boxes_diction = {}
    for box in Boxes:
        Box_diction = []
        name = box[4]
        print("名称", name)
        xmin = int(float(box[0]))
        ymin = int(float(box[1]))
        xmax = xmin + int(float(box[2]))
        ymax = ymin + int(float(box[3]))
        xcenter = (xmax + xmin) / 2
        ycenter = (ymax + ymin) / 2
        Box_diction = [xmin, ymin, xmax, ymax, xcenter, ycenter]
        if name in Boxes_diction:
            Boxes_diction[name].append(Box_diction)
        else:
            Boxes_diction[name] = [Box_diction]

    for key, value in Boxes_diction.items():
        Boxes_diction[key] = np.array(value)
    return Boxes_diction

def Image_preprocessing(img, long_cor, short_cor, step, scaling):
    """
    图像预处理——裁剪和缩放

    Input
    img:输入图像
    long_cor:长边裁剪位置
    short_cor:短边裁剪位置
    step:裁剪长度
    scale1:缩放尺寸

    Return
    img:处理后的图像
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[short_cor: short_cor + step, long_cor: long_cor + step]
    img = cv2.resize(img, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)
    return img

# def Read_Xmp(path_img):
#     """
#     读取图像的xmp信息——经纬高和云台角度
#
#     Input
#     path_img:图像地址
#
#     Return
#     loc:经纬高
#     degree:云台角度
#     """
#     img = Image(path_img)
#     xmp_data = img.read_xmp()
#     xmp_list = []
#     for key, value in xmp_data.items():
#         xmp_list.append((key, value))
#
#     loc = [0] * 3
#     degree = [0] * 3
#     degree_flight = [0] * 3
#     for key, value in xmp_list:
#         if key == 'Xmp.drone-dji.GpsLatitude':
#             loc[0] = float(value)
#         elif key == 'Xmp.drone-dji.GpsLongitude':
#             loc[1] = float(value)
#         elif key == 'Xmp.drone-dji.AbsoluteAltitude':
#             loc[2] = float(value)
#
#         elif key == 'Xmp.drone-dji.GimbalRollDegree':
#             degree[0] = float(value)
#         elif key == 'Xmp.drone-dji.GimbalYawDegree':
#             degree[1] = float(value)
#         elif key == 'Xmp.drone-dji.GimbalPitchDegree':
#             degree[2] = float(value)
#
#         elif key == 'Xmp.drone-dji.FlightRollDegree':
#             degree_flight[0] = float(value)
#         elif key == 'Xmp.drone-dji.FlightYawDegree':
#             degree_flight[1] = float(value)
#         elif key == 'Xmp.drone-dji.FlightPitchDegree':
#             degree_flight[2] = float(value)
#     return np.array(loc), np.array(degree), np.array(degree_flight)

def Box_Obj_Statistics(path_xml):
    """
    提取图像的物料识别结果

    Input
    path_xml:xml文件地址

    Return
    list_obj:物料识别结果字典，每个key中每行为[xmin, ymin, xmax, ymax, xcenter, ycenter]
    """
    # 定义一个字典来存储所有的物料框中心点
    list_obj = {}

    # 解析XML文件
    tree = ET.parse(path_xml)

    root = tree.getroot()

    # 遍历所有的object标签
    for obj in root.findall('object'):
        name = obj.find('name').text
        obj_box = []
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        xcenter = (xmax + xmin) / 2
        ycenter = (ymax + ymin) / 2
        obj_box = [xmin, ymin, xmax, ymax, xcenter, ycenter]

        if name in list_obj:
            list_obj[name].append(obj_box)
        else:
            list_obj[name] = [obj_box]

        # 将列表转换成numpy数组
    for key, value in list_obj.items():
        list_obj[key] = np.array(value)
    return list_obj

def OpticalFlow(img_ref, img_mov, speed_OpticalFlow):
    """
    计算两幅图之间的DIS光流

    Input
    img_ref:参考图
    img_mov:浮动图
    speed_OpticalFlow:光流速度

    Return
    flow:光流图
    """
    hsv = np.zeros_like(img_ref)
    hsv[...,1] = 255
    dis = cv2.DISOpticalFlow_create(speed_OpticalFlow)
    flow = dis.calc(img_ref, img_mov, None, )
    return flow

def Translate_scale(list_objs, long_cor, short_cor, scale1):
    """
    将原始尺寸识别框转换为剪裁后尺寸

    Input
    list_objs:原始尺寸识别框
    long_cor:长边裁剪位置
    short_cor:短边裁剪位置
    scale1:缩放尺寸

    Return
    list_objs_cut:剪裁后尺寸的识别框
    """
    list_objs_cut = {}
    for key, array in list_objs.items():

        list_objs_cut[key] = array.copy()
        list_objs_cut[key][:, 0] = (list_objs[key][:, 0] - long_cor) * scale1
        list_objs_cut[key][:, 2] = (list_objs[key][:, 2] - long_cor) * scale1
        list_objs_cut[key][:, 4] = (list_objs[key][:, 4] - long_cor) * scale1

        # 对第二列、第四列、第六列进行操作
        list_objs_cut[key][:, 1] = (list_objs[key][:, 1] - short_cor) * scale1
        list_objs_cut[key][:, 3] = (list_objs[key][:, 3] - short_cor) * scale1
        list_objs_cut[key][:, 5] = (list_objs[key][:, 5] - short_cor) * scale1

    return list_objs_cut

def Objs_match(list_objs_ref, list_objs_mov, img_ref, img_mov1, flow, obj_name, threshold, scale_obj, visual_option):
    """
    根据两幅图像光流结果进行指定物料的匹配

    Input
    list_objs_ref:参考图的物料识别结果
    list_objs_mov:浮动图的物料识别结果
    img_ref:剪裁后的灰度参考图
    img_mov1:剪裁后的灰度浮动图
    flow:两幅图的光流结果
    obj_name:物料名称
    threshold:光流物料匹配的重合度像素阈值，大于该阈值的认为没有光流结果与之匹配
    scale2:取识别框的中心区域处相对于原始识别框scale2大小的图像进行光流匹配
    visual_option:可视化开关，1表示开，0表示关

    Return
    obj_matched:匹配结果，n×4，[xcenter_ref, ycenter_ref, xcenter_mov, ycenter_mov]
    dist_list:距离矩阵，n×m，位置(n, m)处对应ref图中第n个物料与mov图中第m个物料之间像素距离
    index_list:索引矩阵，第i行的数字j对应ref图中第i个物料匹配mov图中第j个物料
    """
    obj_matched = []
    dist_list = []
    index_list = []

    list_ref = list_objs_ref[obj_name]
    list_mov = list_objs_mov[obj_name]

    img_show = img_ref
    img_show = cv2.cvtColor(img_show, cv2.COLOR_GRAY2BGR)
    # 遍历list_obj_ref中obj_name中的识别框
    for i in range(len(list_ref)):

        xcenter = list_ref[i][4]
        ycenter = list_ref[i][5]
        w2 = (list_ref[i][2] - list_ref[i][0]) * scale_obj
        h2 = (list_ref[i][3] - list_ref[i][1]) * scale_obj
        xmin2 = int(xcenter - w2 / 2)
        xmax2 = int(xcenter + w2 / 2)
        ymin2 = int(ycenter - h2 / 2)
        ymax2 = int(ycenter + h2 / 2)

        # 获取区域内光流
        Vx_left = flow[:, :, 0][ymin2:ymax2, xmin2:xmax2]
        Vy_left = flow[:, :, 1][ymin2:ymax2, xmin2:xmax2]

        # 计算区域内光流的均值
        roi_x = Vx_left
        roi_y = Vy_left
        mean_x = np.mean(roi_x)
        mean_y = np.mean(roi_y)

        # 计算离当前位置最近的物料点
        xcenter2 = int(xcenter + mean_x)
        ycenter2 = int(ycenter + mean_y)
        distances = np.sqrt((list_mov[:, 4] - xcenter2)**2 + (list_mov[:, 5] - ycenter2)**2)

        # 更新距离索引
        dist_list.append(distances.tolist())

        # 找到最小距离对应的索引
        min_distance, index = np.min(distances), np.argmin(distances)

        # 更新位置匹配结果索引
        index_list.append(index)

        if min_distance < threshold:
            # 获取离ref中该物料最近的物料坐标
            nearest_coordinate = list_mov[index,4:6]

        else:
            # 对于没有光流匹配的位置，输出像素位置[0, 0]
            nearest_coordinate = [0, 0]
        obj_matched.append([xcenter, ycenter, nearest_coordinate[0], nearest_coordinate[1]])
        # 画出物料框
        if visual_option == 1:
            cv2.rectangle(img_show, (xmin2, ymin2), (xmax2, ymax2), (0, 255, 0), 2)
            for i in range(xmin2, xmax2, 5):
                for j in range(ymin2, ymax2, 5):

                    cv2.arrowedLine(img_show, (i, j), (int(i + flow[i][j][0]), int(j + flow[i][j][1])), color=(255, 255, 0), thickness=1)

    if visual_option == 1:

        cv2.imshow('Image with box', img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return np.array(obj_matched), np.array(dist_list), np.array(index_list).reshape(-1, 1)

def Restore_match(obj_matched, long_cor, short_cor, scale1):
    """
    将在剪裁尺寸下获取的物料匹配结果转换到原始尺寸中

    Input
    obj_matched:剪裁尺寸的物料匹配结果
    long_cor:长边裁剪位置
    short_cor:短边裁剪位置
    scale1:缩放尺寸

    Return
    obj_matched_ori:原始尺寸的物料匹配结果
    """
    obj_matched_ori = obj_matched.copy()
    for i in range(obj_matched_ori.shape[0]):
        obj_matched_ori[i, 0] = obj_matched[i, 0] / scale1 + long_cor
        obj_matched_ori[i, 1] = obj_matched[i, 1] / scale1 + short_cor
        if obj_matched_ori[i, 2]*obj_matched_ori[i, 3] != 0:
            obj_matched_ori[i, 2] = obj_matched[i, 2] / scale1 + long_cor
            obj_matched_ori[i, 3] = obj_matched[i, 3] / scale1 + short_cor

    return obj_matched_ori

def Matched_Derepeat(
        materials_matched1_ori, list_materials_ref_cut, list_materials_mov_cut, index_list, dist_list, obj_name, long_cor, short_cor, scale1, threshold):
    """
    对光流匹配后的结果进行匈牙利去重

    Return
    materials_matched_derepeat:[ref图物料像素中心x坐标, ref图物料像素中心y坐标, mov图物料像素中心x坐标, mov图物料像素中心y坐标]

    """
    list_materials_ref = list_materials_ref_cut[obj_name]
    list_materials_mov = list_materials_mov_cut[obj_name]
    materials_matched_derepeat = materials_matched1_ori.copy()

    num = 0
    alpha = 10000   # 设置一个很大值来占位重复绝缘子的索引数组

    while True:
        num += 1

        flag = False    # 初始化循环终止条件flag，当flag为true时表示有重复，中断循环

        # 循环判断index_list有无重复
        for i in range(len(index_list)):

            # 只有i非最后一位才需要判断
            if i < len(index_list) - 1:

                # 循环判断第i位后面的index是否与index(i)重复，也即是参考图中是否有两个不同绝缘子匹配到浮动图上同一绝缘子
                for j in range(i + 1, len(index_list)):

                    # 判断匹配是否重复
                    if index_list[i] == index_list[j]:
                        repeat_index = [i, j]   # 记录匹配重复的物料在ref图中的哪个位置

                        flag = True
                        break

            # 若flag为true，发现重复，中止判断，开始处理
            if flag:
                break

        # 当没有发现重复匹配时，退出while循环
        if not flag:
            break

        # 当第一个重复物料比第二个重复物料更近时
        if dist_list[repeat_index[0], index_list[repeat_index[0]]] < dist_list[repeat_index[1], index_list[repeat_index[1]]]:

            sorted_arr = np.sort(dist_list[repeat_index[1], :]) # 对第二个物料的距离列表进行升序排列

            # 找到未进行去重时，败方物料所匹配的物料距离在升序距离列表中的索引
            # this_min_index = np.argwhere(sorted_arr[repeat_index[1], :] == dist_list[repeat_index[1], index_list[repeat_index[1]]])
            this_min_index = np.where(sorted_arr == dist_list[repeat_index[1], index_list[repeat_index[1]]])[0]

            # 当没有超出索引范围时
            if this_min_index < len(list_materials_mov) - 1:
                next_min_index = this_min_index + 1

                new_index = np.where(dist_list[repeat_index[1], :] == sorted_arr[next_min_index])[0]    # 找到次优解在dist_list的索引

                # 当次优解依然满足设置的阈值时，用次优解更新index_list和materials_matched_derepeat
                if sorted_arr[next_min_index] < threshold:

                    index_list[repeat_index[1]] = new_index # 更新index_list
                    materials_matched_derepeat[repeat_index[1], 2:4] = [
                        list_materials_ref[new_index, 0] / scale1 + long_cor,
                        list_materials_ref[new_index, 1] / scale1 + short_cor
                    ]
                else:
                    index_list[repeat_index[1]] = alpha
                    alpha += 1  # 将alpha＋1处理，防止出现同样的alpha

                    materials_matched_derepeat[repeat_index[1], 2:4] = [0, 0]
            else:
                index_list[repeat_index[1]] = alpha
                alpha += 1  # 将alpha＋1处理，防止出现同样的alpha
                materials_matched_derepeat[repeat_index[1], 2:4] = [0, 0]

        # 当第二个重复物料比第一个重复物料更近时
        else:
            sorted_arr = np.sort(dist_list[repeat_index[0], :]) # 对第二个物料的距离列表进行升序排列

            # 找到未进行去重时，败方物料所匹配的物料距离在升序距离列表中的索引
            this_min_index = np.where(sorted_arr == dist_list[repeat_index[0], index_list[repeat_index[0]]])[0]

            # 当没有超出索引范围时
            if this_min_index < len(list_materials_mov) - 1:
                next_min_index = this_min_index + 1

                new_index = np.where(dist_list[repeat_index[0], :] == sorted_arr[next_min_index])[0]    # 找到次优解在dist_list的索引

                # 当次优解依然满足设置的阈值时，用次优解更新index_list和materials_matched_derepeat
                if sorted_arr[next_min_index] < threshold:

                    index_list[repeat_index[0]] = new_index # 更新index_list
                    materials_matched_derepeat[repeat_index[0], 2:4] = [
                        list_materials_ref[new_index, 0] / scale1 + long_cor,
                        list_materials_ref[new_index, 1] / scale1 + short_cor
                    ]
                else:
                    index_list[repeat_index[0]] = alpha
                    alpha += 1  # 将alpha＋1处理，防止出现同样的alpha
                    materials_matched_derepeat[repeat_index[0], 2:4] = [0, 0]
            else:
                index_list[repeat_index[0]] = alpha
                alpha += 1  # 将alpha＋1处理，防止出现同样的alpha
                materials_matched_derepeat[repeat_index[0], 2:4] = [0, 0]

    return materials_matched_derepeat

def WGS84_to_ENU(Loc0, loc):

    [ecef_x0, ecef_y0, ecef_z0] = WGS84_to_ECEF(Loc0)
    [ecef_x1, ecef_y1, ecef_z1] = WGS84_to_ECEF(loc)
    offset_x, offset_y, offset_z = ecef_x1 - ecef_x0, ecef_y1 - ecef_y0, ecef_z1 - ecef_z0

    lat0 = math.radians(Loc0[0])
    lon0 = math.radians(Loc0[1])

    cosLat = math.cos(lat0)
    cosLon = math.cos(lon0)
    sinLat = math.sin(lat0)
    sinLon = math.sin(lon0)


    y = -1 * sinLon *offset_x + cosLon * offset_y
    x = -1 * sinLat * cosLon * offset_x - 1 * sinLat *sinLon * offset_y + cosLat * offset_z
    z = cosLat * cosLon * offset_x + cosLat * sinLon * offset_y + sinLat * offset_z

    xyz = [x, y, z]
    return np.array(xyz)

def WGS84_to_ECEF(XYZ):
    long_r = 6378137.0000
    shot_r = 6356752.3142
    oblateness = (long_r - shot_r) / long_r # 扁率
    pow_e_2 = oblateness * (2.0 - oblateness) # 第一偏心率的平方

    # 将WGS84转为ECEF
    lat = math.radians(float(XYZ[0]))
    lon = math.radians(float(XYZ[1]))
    alt = float(XYZ[2])

    cosLon = math.cos(lon)
    cosLat = math.cos(lat)
    sinLon = math.sin(lon)
    sinLat = math.sin(lat)

    N = long_r / math.sqrt(1.0 - pow_e_2 * sinLat * sinLat) # 卯酉圈曲率半径

    ecef_x = (N + alt) * cosLat * cosLon
    ecef_y = (N + alt) * cosLat * sinLon
    ecef_z = (N * (1.0 - pow_e_2) + alt) * sinLat
    return (ecef_x, ecef_y, ecef_z)

def Pixel_to_angle(pixel_x, pixel_y, x_resolution, y_resolution, d):
    """
    将像素坐标转换为相对于相机的角度脱靶量
    """
    pixel_x = pixel_x - x_resolution / 2
    pixel_y = y_resolution / 2  - pixel_y   # 因为像素坐标系的y向（向下为正）跟相机像素坐标系的z向（向上为正）方向相反
    # 相机中心到像素坐标系的距离d，计算过程见PPT
    # d = 2903
    # 物料向量模长l
    l = math.sqrt(d ** 2 + pixel_x ** 2 + pixel_y ** 2)
    vec = [0, 0, 0]
    vec[0] = d / l
    vec[1] = pixel_x / l
    vec[2] = pixel_y / l

    return vec

def Camera_vec_to_world_vec(vec, roll, yaw, pitch):

    # 射线L的方向向量
    vec = np.array(vec).reshape(3, 1)
    yaw = np.deg2rad(yaw) # 绕z轴左手系旋转
    pitch = np.deg2rad(-pitch)   # 绕y轴左手旋转，由于云台是右手系，所以需要前面加负号
    # 计算yaw旋转矩阵 Rz(yaw)
    Rz_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])

    # 计算pitch旋转矩阵 Ry(pitch)
    Ry_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                         [0, 1, 0],
                         [-np.sin(pitch), 0, np.cos(pitch)]])

    Rx_roll = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])

    # # 计算旋转矩阵 RYZ，先转Y后转Z，绕动轴
    # RYZ = np.dot(Rz_yaw, Ry_pitch)
    # 计算旋转矩阵 RXYZ，先转X、Y后转Z，绕动轴
    RXYZ = np.dot(Rz_yaw, Ry_pitch, Rx_roll)
    # 计算在坐标系B中的方向向量v
    vec_world = np.dot(RXYZ, vec)
    vec_world = vec_world.flatten()
    return vec_world

def Calculate_point_in_L_and_norm(v1, p1, v2, p2):
    """
    根据空间射线和无人机坐标计算物料空间点落在第一条空间射线上的坐标
    """
    normal_vector = np.cross(v1, v2)
    if (normal_vector == 0).all():
        point_in_L_and_norm = np.array([0, 0, 0])
    else:
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        a1 = p1[0] - p2[0]
        a2 = p1[1] - p2[1]
        a3 = p1[2] - p2[2]
        b1 = v2[1] * normal_vector[2] -v2[2] * normal_vector[1]
        b2 = v2[2] * normal_vector[0] -v2[0] * normal_vector[2]
        b3 = v2[0] * normal_vector[1] -v2[1] * normal_vector[0]
        t = - (a1 * b1 + a2 * b2 + a3 * b3) / (v1[0] * b1 + v1[1] * b2 + v1[2] * b3)
        point_in_L_and_norm = np.array([p1[0] + t * v1[0], p1[1] + t * v1[1], p1[2] + t * v1[2]])
    return point_in_L_and_norm

def Calculate_3d_pts(center_obj_mov1, center_obj_mov2, degree_mov1, degree_mov2, loc_mov1, loc_mov2, x_resolution, y_resolution, d):
    """
    根据物料的像素坐标计算物料的空间坐标

    Return
    obj_3d_pts:[x, y, z, pix_x_mov1, pix_y_mov1, pix_x_mov2, pix_y_mov2, j]
    """

    obj_3d_pts = []
    j = 0
    for i in range(len(center_obj_mov1)):
        if center_obj_mov1[i][2] * center_obj_mov2[i][2] != 0:
            vec1 = Pixel_to_angle(center_obj_mov1[i, 2], center_obj_mov1[i, 3], x_resolution, y_resolution, d)
            vec_world1 = Camera_vec_to_world_vec(vec1, degree_mov1[0], degree_mov1[1], degree_mov1[2])

            vec2 = Pixel_to_angle(center_obj_mov2[i, 2], center_obj_mov2[i, 3], x_resolution, y_resolution, d)

            vec_world2 = Camera_vec_to_world_vec(vec2, degree_mov2[0], degree_mov2[1], degree_mov2[2])

            point_in_L1_and_norm = Calculate_point_in_L_and_norm(vec_world1, loc_mov1, vec_world2, loc_mov2)
            point_in_L2_and_norm = Calculate_point_in_L_and_norm(vec_world2, loc_mov2, vec_world1, loc_mov1)
            point_center = (point_in_L1_and_norm + point_in_L2_and_norm) / 2

            point_center_rounded = np.round(point_center, 3)
            # [x, y, z, pix_x1, pix_y1, pix_x2, pix_y2, j]
            value = [point_center_rounded[0], point_center_rounded[1], point_center_rounded[2],
                     center_obj_mov1[i, 2], center_obj_mov1[i, 3], center_obj_mov2[i, 2], center_obj_mov2[i, 3], int(j)]
            obj_3d_pts.append(value)
            j += 1

    return np.array(obj_3d_pts)



