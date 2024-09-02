import numpy as np
import cv2
import os
import json
import math

def Cor2Result(json1):
    # def cor2Result(Result1, Result2, loc0, loc_ganta, degree_front, degree_back):

    # 将json格式转换为Python格式
    data = json.loads(json1)

    # 提取物料列表
    Obj_3d_pts1 = data["wuliaodata"][0]
    Obj_3d_pt2 = data["wuliaodata"][1]
    # 将列表转为字典格式
    obj_3d_pts1 = {k: np.array(v) for k, v in Obj_3d_pts1.items()}
    obj_3d_pt2 = {k: np.array(v) for k, v in Obj_3d_pt2.items()}

    # obj_3d_pts1 = {
    #     'jueyuanzi': np.array([
    #         [ 5.8020e+00,  5.6230e+00, -7.6390e+00,  1.7940e+03,  1.5865e+03,
    #           1.7495e+03,  1.5515e+03,  0.0000e+00,  0.0000e+00],
    #         [ 6.1050e+00,  6.4590e+00, -7.4260e+00,  2.0595e+03,  1.4785e+03,
    #           2.0230e+03,  1.4780e+03,  1.0000e+00,  0.0000e+00],
    #         [ 6.5800e+00,  7.3020e+00, -7.6790e+00,  2.3150e+03,  1.4170e+03,
    #           2.3000e+03,  1.4500e+03,  2.0000e+00,  0.0000e+00],
    #         [ 6.2500e+00,  5.4830e+00, -7.4400e+00,  1.7590e+03,  1.4345e+03,
    #           1.7370e+03,  1.4000e+03,  3.0000e+00,  0.0000e+00],
    #         [ 6.6310e+00,  5.3410e+00, -7.4930e+00,  1.7285e+03,  1.3545e+03,
    #           1.7255e+03,  1.3125e+03,  4.0000e+00,  0.0000e+00],
    #         [ 6.6810e+00,  6.8410e+00, -7.5310e+00,  2.1795e+03,  1.3625e+03,
    #           2.1695e+03,  1.3765e+03,  5.0000e+00,  0.0000e+00],
    #         [ 7.0280e+00,  7.1550e+00, -7.4550e+00,  2.2745e+03,  1.2685e+03,
    #           2.2800e+03,  1.2990e+03,  6.0000e+00,  0.0000e+00],
    #         [ 6.8630e+00,  7.1940e+00, -6.7850e+00,  2.3020e+03,  1.1725e+03,
    #           2.3000e+03,  1.2005e+03,  7.0000e+00,  0.0000e+00],
    #         [ 7.1220e+00,  6.5960e+00, -6.5340e+00,  2.1165e+03,  1.0525e+03,
    #           2.1280e+03,  1.0570e+03,  8.0000e+00,  0.0000e+00],
    #         [ 7.5220e+00,  5.9650e+00, -6.8320e+00,  1.9265e+03,  1.0230e+03,
    #           1.9585e+03,  1.0110e+03,  9.0000e+00,  0.0000e+00]
    #     ]),
    #     'ganta': np.array([
    #         [ -6.02242466, -13.14875747,   7.12633281,   0.        ],
    #         [ -5.8764968 , -12.81830249,   6.9418057 ,   0.        ],
    #         [ -5.94422549, -12.95062256,   7.00639707,   0.        ]
    #     ]),
    #     'hengdan': np.array([
    #         [ 6.6960e+00,  6.3280e+00, -6.7670e+00,  2.0255e+03,  1.1980e+03,
    #           2.0175e+03,  1.1985e+03,  0.0000e+00,  0.0000e+00],
    #         [ 6.6610e+00,  6.3280e+00, -7.6490e+00,  2.0255e+03,  1.3850e+03,
    #           2.0165e+03,  1.3795e+03,  1.0000e+00,  0.0000e+00]
    #     ])
    # }

    # obj_3d_pts2 = {
    #     'jueyuanzi': np.array([
    #         [ 5.8020e+00,  5.6230e+00, -7.6390e+00,  1.7940e+03,  1.5865e+03, 1.7495e+03,  1.5515e+03,  0.0000e+00,  0.0000e+00],
    #         [ 6.1050e+00,  6.4590e+00, -7.4260e+00,  2.0595e+03,  1.4785e+03, 2.0230e+03,  1.4780e+03,  1.0000e+00,  0.0000e+00],
    #         [ 6.5800e+00,  7.3020e+00, -7.6790e+00,  2.3150e+03,  1.4170e+03, 2.3000e+03,  1.4500e+03,  2.0000e+00,  0.0000e+00],
    #         [ 6.2500e+00,  5.4830e+00, -7.4400e+00,  1.7590e+03,  1.4345e+03, 1.7370e+03,  1.4000e+03,  3.0000e+00,  0.0000e+00],
    #         [ 6.6310e+00,  5.3410e+00, -7.4930e+00,  1.7285e+03,  1.3545e+03, 1.7255e+03,  1.3125e+03,  4.0000e+00,  0.0000e+00],
    #         [ 6.6810e+00,  6.8410e+00, -7.5310e+00,  2.1795e+03,  1.3625e+03, 2.1695e+03,  1.3765e+03,  5.0000e+00,  0.0000e+00],
    #         [ 7.0280e+00,  7.1550e+00, -7.4550e+00,  2.2745e+03,  1.2685e+03, 2.2800e+03,  1.2990e+03,  6.0000e+00,  0.0000e+00],
    #         [ 6.8630e+00,  7.1940e+00, -6.7850e+00,  2.3020e+03,  1.1725e+03, 2.3000e+03,  1.2005e+03,  7.0000e+00,  0.0000e+00],
    #         [ 7.1220e+00,  6.5960e+00, -6.5340e+00,  2.1165e+03,  1.0525e+03, 2.1280e+03,  1.0570e+03,  8.0000e+00,  0.0000e+00],
    #         [ 7.5220e+00,  5.9650e+00, -6.8320e+00,  1.9265e+03,  1.0230e+03, 1.9585e+03,  1.0110e+03,  9.0000e+00,  0.0000e+00]
    #     ]),
    #     'ganta': np.array([
    #         [ -6.02242466, -13.14875747,   7.12633281,   0.        ],
    #         [ -5.8764968 , -12.81830249,   6.9418057 ,   0.        ],
    #         [ -5.94422549, -12.95062256,   7.00639707,   0.        ]
    #     ]),
    #     'hengdan': np.array([
    #         [ 6.6960e+00,  6.3280e+00, -6.7670e+00,  2.0255e+03,  1.1980e+03, 2.0175e+03,  1.1985e+03,  0.0000e+00,  0.0000e+00],
    #         [ 6.6610e+00,  6.3280e+00, -7.6490e+00,  2.0255e+03,  1.3850e+03, 2.0165e+03,  1.3795e+03,  1.0000e+00,  0.0000e+00]
    #     ])
    # }

    obj_3d_pts1 = Add_order(obj_3d_pts1, 0)
    obj_3d_pts2 = Add_order(obj_3d_pt2, 1)

    # 坐标系原点
    # loc0 = [30.3067, 114.4250, 38.7490]
    # loc0 = np.array(loc0)
    loc0 = [data["startLoc"]["lat"], data["startLoc"]["lon"], data["startLoc"]["alt"]]
    # 杆塔坐标
    loc_ganta = [data["towerLoc"]["lat"], data["towerLoc"]["lon"], data["towerLoc"]["alt"]]
    xyz_ganta = WGS84_to_ENU(loc0, loc_ganta)
    xyz_ganta = [ 6.6,  7, -7]

    # 本级杆塔的导线方向
    # degree_front = np.array([0, 161.3, -54.4])
    degree_front = [data["degree_front"]["roll"], data["degree_front"]["yaw"], data["degree_front"]["pitch"]]
    # 上一级杆塔的导线方向
    # degree_back = np.array([0, 32.1, -54.4])
    degree_back = [data["degree_back"]["roll"], data["degree_back"]["yaw"], data["degree_back"]["pitch"]]
    '''
    可调参数
    '''
    range_left = -0.6   # 点云匹配左边界
    range_right = 0.6   # 点云匹配左边界
    step = 0.05         # 点云匹配步进
    threshold = 0.5     # 距离阈值，大于该阈值的为独立物料
    threshold2 = 0.3    # 距离阈值，小于该阈值的认为是相同物料
    threshold3 = 0.3    # 避雷器和熔断器的去重距离阈值，小于该阈值的为相同物料
    threshold4 = 0.2    # 与横担关联的绝缘子在横担下方极限距离
    threshold5 = 0.4    # 与横担关联的绝缘子在横担上方极限距离
    threshold_angle = 10# 绝缘子方向与横担方向夹角阈值

    '''
    绝缘子点云配准
    '''
    # 从字典里将绝缘子的三维点提取出来
    # jyz_3d_pts1为[x, y, z, mov1中x_pixel, mov2中y_pixel, 序号]
    jyz_3d_pts1 = obj_3d_pts1['jueyuanzi'].astype(np.float32)

    jyz_3d_pts2 = obj_3d_pts2['jueyuanzi'].astype(np.float32)
    # 粗配准，输出粗配准后的第二幅图的物料三维点
    jyz_3d_pts2_CM, T1 = coarse_matching(jyz_3d_pts1, jyz_3d_pts2, range_left, range_right, step, threshold)

    # 精配准
    jyz_3d_pts_FM_rep, jyz_3d_pts2_FM_rep, T2 = fine_matching(jyz_3d_pts1, jyz_3d_pts2_CM, threshold, threshold2)

    """
    绝缘子物料统计
    """
    T = T1 + T2
    # group_jyz_pts = [x, y, z, pix_x_mov1, pix_y_mov1, pix_x_mov2, pix_y_mov2, j]
    index_jyz, group_jyz_pts = Find_duplicate_jyz(jyz_3d_pts1, jyz_3d_pts2, jyz_3d_pts_FM_rep, jyz_3d_pts2_FM_rep, T)
    print("绝缘子数量：", len(group_jyz_pts))
    """
    避雷器、熔断器统计
    """
    # 当检测到避雷器时
    if 'bileiqi' in obj_3d_pts1 and 'bileiqi' in obj_3d_pts2:
        # 对避雷器进行去重，并统计数量
        num_blq, _ = Calculate_obj_num(obj_3d_pts1, obj_3d_pts2, 'bileiqi', T, threshold, threshold2, threshold3)
    else:
        num_blq = 0

    # 当检测到熔断器时
    if 'rongduanqi' in obj_3d_pts1 and 'rongduanqi' in obj_3d_pts2:
        # 对熔断器进行去重，并统计数量
        num_rdq, _ = Calculate_obj_num(obj_3d_pts1, obj_3d_pts2, 'rongduanqi', T, threshold, threshold2, threshold3)
    else:
        num_rdq = 0

    """
    横担数量统计
    """
    # 选择两个拍摄点位中横担数量最多的位置
    if 'hengdan' in obj_3d_pts1 and 'hengdan' in obj_3d_pts2:
        # 对横担进行去重，并统计数量
        num_hd, hd_3d_pts = Calculate_obj_num(obj_3d_pts1, obj_3d_pts2, 'hengdan', T, threshold, threshold2, threshold3)
    else:
        num_hd = 0

    """
    横担间距计算
    """
    # 将横担高度按升序排列
    hengdan_h = np.sort(hd_3d_pts[:, 2]).reshape(-1, 1)
    jyznum_in_hengda = np.zeros((len(hengdan_h), 1))

    gap_hengdan_h = np.diff(hengdan_h)

    """
    杆塔高度计算
    """
    ganta1 = obj_3d_pts1['ganta']   # 每一行为[杆塔顶端，杆塔底部，杆塔高度]
    ganta2 = obj_3d_pts2['ganta']
    ganta12 = np.vstack((ganta1, ganta2))
    ganta = np.median(ganta12, axis=0)
    ganta_bottom = ganta[1]
    ganta_h = ganta[2]

    """
    变压器高度计算
    """
    if 'bianyaqi' in obj_3d_pts1 and 'bianyaqi' in obj_3d_pts2:
        byq_3d_pts1 = obj_3d_pts1['bianyaqi']
        byq_3d_pts2 = obj_3d_pts2['bianyaqi']
        byq_h = (byq_3d_pts1[2] + byq_3d_pts2[2]) / 2 - ganta_bottom

    """
    导线间距计算
    """
    # 确定导线前方向与后方向
    direction_front_line_2d = Calculate_direction(degree_front)
    direction_back_line_2d = Calculate_direction(degree_back)

    # 确定横担上的绝缘子位置
    group_jyz_layer = []
    for i in range(len(hengdan_h)):

        # 统计一层横担上的绝缘子
        h = hengdan_h[i]
        jyz_layer = []
        for j in range(len(group_jyz_pts)):

            # 当绝缘子的高度上距离横担足够近时
            if h - threshold4 < group_jyz_pts[j, 2] < h + threshold5:
                jyz_layer.append(group_jyz_pts[j])

        # 当该层有绝缘子时
        if jyz_layer != []:

            # 计算绝缘子与横担的距离
            jyz_layer2 = Calculate_distance_jyz_hd(jyz_layer, direction_front_line_2d, xyz_ganta)

            # 当该层只有一个绝缘子时
            if len(jyz_layer) == 1:
                jyznum_in_hengda[i] = 1

            # 当该层不止一个绝缘子时
            else:

                # 挑选两个沿着导线方向离横担最远的绝缘子的方向direction_jyz
                direction_jyz = Calculate_jyz_direction(jyz_layer2)

                # 横担方向
                direction_hengdan = np.array([- direction_front_line_2d[1], direction_front_line_2d[0]])

                # 计算绝缘子方向与横担方向的夹角
                delta_degree = Calculate_delta_degree(direction_jyz, direction_hengdan)

                # 当夹角较小时，表示这个横担是前向导线的横担
                if delta_degree < threshold_angle:

                    # 筛选绝缘子直线上的绝缘子
                    jyz_layer_on_line = Calculate_jyz_on_line(jyz_layer2, direction_jyz)

                    # 将绝缘子按照绝缘子方向排列
                    jyz_layer_on_line2 = sort_points_by_direction(jyz_layer_on_line, direction_jyz)

                    # 计算绝缘子之间的间距
                    points_array = np.array([point[:2] for point in jyz_layer_on_line2])

                    # 计算相邻点之间的距离
                    distances = []
                    for i in range(len(points_array) - 1):
                        dist = np.linalg.norm(points_array[i] - points_array[i + 1])
                        distances.append(dist)

    return obj_3d_pts1

def Add_order(data_dict, order):
    # 遍历字典中的每个键值对
    for key in data_dict:
        # 获取原数组
        original_array = data_dict[key]

        if order == 0:
            # 创建一个与原数组行数相同、列数为1的零数组
            zero_column = np.zeros((original_array.shape[0], 1), dtype=original_array.dtype)
        elif order == 1:
            zero_column = np.ones((original_array.shape[0], 1), dtype=original_array.dtype)
        # 将原数组与零列水平拼接
        updated_array = np.hstack((original_array, zero_column))

        # 更新字典中的数组
        data_dict[key] = updated_array
    return data_dict


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

def coarse_matching(xx_3d_pts, xx_3d_pts2, range_left, range_right, step, threshold):
    """
    Return
    obj_3d_pts2_CM:[x, y, z, pix_x_mov1, pix_y_mov1, pix_x_mov2, pix_y_mov2, j]
    """

    weight = 0  # 记录权重
    T1 = np.zeros(3)  # 初始化变换向量

    m_values = np.arange(range_left, range_right + step, step)
    n_values = np.arange(range_left, range_right + step, step)
    l_values = np.arange(range_left, range_right + step, step)

    # 遍历变换向量可能的值
    for m in m_values:
        for n in n_values:
            for l in l_values:
                translation_vector = np.array([m, n, l])
                points2_t = xx_3d_pts2[:, :3] + translation_vector

                dist_pointpair = []
                vec_pointpair = []

                # 计算每个points1中点到points2_t中点的最小距离
                for i in range(xx_3d_pts.shape[0]):
                    distances = np.sqrt(np.sum((points2_t - xx_3d_pts[i, :3])**2, axis=1))
                    min_distance = np.min(distances)
                    index = np.argmin(distances)

                    # 如果距离小于阈值，则记录下来
                    if min_distance < threshold:
                        dist_pointpair.append(min_distance)
                        vec_pointpair.append(xx_3d_pts[i, :3] - points2_t[index])

                # 计算权重
                dist_pointpair = np.array(dist_pointpair)
                weight1 = np.sum(1.0 / (dist_pointpair + 0.02)**1.5)

                # 更新权重和变换向量
                if weight1 >= weight:
                    weight = weight1
                    T1 = translation_vector

    # 应用最佳变换向量
    xx_3d_pts2_CM = xx_3d_pts2.copy()
    xx_3d_pts2_CM[:, :3] += T1

    return xx_3d_pts2_CM, T1

def fine_matching(xx_3d_pts, xx_3d_pts2_CM, threshold, threshold2):
    """
    obj_3d_pts_FM_rep, obj_3d_pts2_FM_rep:[x, y, z, pix_x_mov1, pix_y_mov1, pix_x_mov2, pix_y_mov2, j]
    """

    # 1. 判断明显相距很远的点并剔除
    n = 0
    for j in range(xx_3d_pts.shape[0]):
        j -= n
        distances = np.sqrt(np.sum((xx_3d_pts2_CM[:, :3] - xx_3d_pts[j, :3]) ** 2, axis=1))
        min_distance = np.min(distances)

        if min_distance > threshold:
            xx_3d_pts = np.delete(xx_3d_pts, j, axis=0)
            n += 1

    n = 0
    for j in range(xx_3d_pts2_CM.shape[0]):
        j -= n
        distances = np.sqrt(np.sum((xx_3d_pts[:, :3] - xx_3d_pts2_CM[j, :3]) ** 2, axis=1))
        min_distance = np.min(distances)

        if min_distance > threshold:
            xx_3d_pts2_CM = np.delete(xx_3d_pts2_CM, j, axis=0)
            n += 1

    # 2. 迭代融合重合点
    num = 0
    mean_distance0 = [0, 0]
    while True:
        num += 1

        # 计算points1中每个点的最近点均值
        sum_distance1 = 0
        for j in range(xx_3d_pts.shape[0]):
            distances = np.sqrt(np.sum((xx_3d_pts2_CM[:, :3] - xx_3d_pts[j, :3]) ** 2, axis=1))
            min_distance = np.min(distances)
            sum_distance1 += min_distance
        mean_distance1 = sum_distance1 / xx_3d_pts.shape[0]

        # 识别points1中的唯一点
        # 将离points1中与points2最近距离大于2*mean_distance的points1点剔除
        n = 0
        for j in range(xx_3d_pts.shape[0]):
            j -= n
            distances = np.sqrt(np.sum((xx_3d_pts2_CM[:, :3] - xx_3d_pts[j, :3]) ** 2, axis=1))
            min_distance = np.min(distances)
            if min_distance > threshold2:
                xx_3d_pts = np.delete(xx_3d_pts, j, axis=0)
                n += 1

        # 计算points2中每个点的最近点均值
        sum_distance2 = 0
        for j in range(xx_3d_pts2_CM.shape[0]):
            distances = np.sqrt(np.sum((xx_3d_pts[:, :3] - xx_3d_pts2_CM[j, :3]) ** 2, axis=1))
            min_distance = np.min(distances)
            sum_distance2 += min_distance
        mean_distance2 = sum_distance2 / xx_3d_pts2_CM.shape[0]

        # 将离points2中与points1最近距离大于threshold2的points2点剔除
        n = 0
        for j in range(xx_3d_pts2_CM.shape[0]):
            j -= n
            distances = np.sqrt(np.sum((xx_3d_pts[:, :3] - xx_3d_pts2_CM[j, :3]) ** 2, axis=1))
            min_distance = np.min(distances)
            if min_distance > threshold2:
                xx_3d_pts2_CM = np.delete(xx_3d_pts2_CM, j, axis=0)
                n += 1

        # 设置循环终止条件，当meandistance不再变化时
        if abs(mean_distance0[0] - mean_distance1) < 0.001 and abs(mean_distance0[1] - mean_distance2) < 0.001:
            print('配准过程满足第一个终止条件')
            break

        # 设置循环终止条件，循环10次时
        if num == 10:
            print('配准过程满足第二个终止条件')
            break

        # 更新循环条件
        mean_distance0 = [mean_distance1, mean_distance2]

    # 更新points1、points2
    centroid1 = np.mean(xx_3d_pts[:, :3], axis=0)
    centroid2 = np.mean(xx_3d_pts2_CM[:, :3], axis=0)
    T2 = centroid1 - centroid2

    xx_3d_pts_FM_rep = xx_3d_pts
    xx_3d_pts2_FM_rep = xx_3d_pts2_CM
    xx_3d_pts2_FM_rep[:, :3] += T2

    return xx_3d_pts_FM_rep, xx_3d_pts2_FM_rep, T2


def Find_duplicate_jyz(jyz_3d_pts, jyz_3d_pts2, jyz_3d_pts_FM_rep, jyz_3d_pts2_FM_rep, T):
    '''
    根据原始点云以及精匹配后的点云进行最终的绝缘子点云统计去重

    Return
    index_1_rep:mov1中与mov2中重复的绝缘子索引，eg:[False, False, False, False,  True,  True,  True,  True,  True, True]
    index_1_unique:mov1中唯一的绝缘子索引
    index_2_unique]:mov2中唯一的绝缘子索引
    group_obj:统计出的绝缘子坐标
    '''
    numbers1 = np.arange(0, jyz_3d_pts.shape[0])
    numbers2 = np.arange(0, jyz_3d_pts2.shape[0])
    index_unique1 = np.setdiff1d(numbers1, jyz_3d_pts_FM_rep[:, 7])
    index_unique2 = np.setdiff1d(numbers2, jyz_3d_pts2_FM_rep[:, 7])

    index_repeat = []  # [第一组图的索引，第二组图的索引]
    for j in range(jyz_3d_pts_FM_rep.shape[0]):
        distances = np.sqrt(
            (jyz_3d_pts2_FM_rep[:, 0] - jyz_3d_pts_FM_rep[j, 0]) ** 2 +
            (jyz_3d_pts2_FM_rep[:, 1] - jyz_3d_pts_FM_rep[j, 1]) ** 2 +
            (jyz_3d_pts2_FM_rep[:, 2] - jyz_3d_pts_FM_rep[j, 2]) ** 2
        )
        index = np.argmin(distances)
        index_repeat.append([jyz_3d_pts_FM_rep[j, 7], jyz_3d_pts2_FM_rep[index, 7]])
    index_repeat = np.array(index_repeat)

    # 将第二组进行平移
    points2_align = jyz_3d_pts2.copy()
    points2_align[:, :3] += T

    # 设置第一、二组数据显示的索引
    index_1_rep = np.zeros(jyz_3d_pts.shape[0], dtype=bool)
    index_1_unique = np.zeros(jyz_3d_pts.shape[0], dtype=bool)
    index_2_unique = np.zeros(points2_align.shape[0], dtype=bool)

    # 设置第一组中重复、唯一索引
    index_1_rep[index_repeat[:, 0].astype(int)] = True
    index_1_unique[index_unique1.astype(int)] = True

    # 设置第二组中唯一索引
    index_2_unique[index_unique2.astype(int)] = True

    # 对唯一点和重合点计数
    count_index_1_repeat = np.sum(index_1_rep)
    count_index_1_unique = np.sum(index_1_unique)
    count_index_2_unique = np.sum(index_2_unique)
    count_sum = count_index_1_repeat + count_index_1_unique + count_index_2_unique

    # pts_1_rep = jyz_3d_pts[index_1_rep, 0:3]
    # pts_1_unique = jyz_3d_pts[index_1_unique, 0:3]
    # pts_2_unique = jyz_3d_pts2[index_2_unique, 0:3]

    pts_1_rep = jyz_3d_pts[index_1_rep]
    pts_1_unique = jyz_3d_pts[index_1_unique]
    pts_2_unique = jyz_3d_pts2[index_2_unique]

    # group_obj = np.vstack((pts_1_rep[:, :3], pts_1_unique[:, :3], pts_2_unique[:, :3]))
    group_obj = np.vstack((pts_1_rep, pts_1_unique, pts_2_unique))


    return [index_1_rep, index_1_unique, index_2_unique], group_obj

def Calculate_obj_num(obj_3d_pts1, obj_3d_pts2, name_obj, T, threshold, threshold2, threshold3):

    # 从字典里将指定物料的三维点提取出来
    xx_3d_pts1 = obj_3d_pts1[name_obj].astype(np.float32)
    xx_3d_pts2 = obj_3d_pts2[name_obj].astype(np.float32)

    # 将第二组平移到与第一组一样的位置
    xx_3d_pts2_FM = xx_3d_pts2.copy()
    xx_3d_pts2_FM[:, :3] = xx_3d_pts2[:, :3] + T
    xx_3d_pts1_rep, xx_3d_pts2_rep, _ = fine_matching(xx_3d_pts1, xx_3d_pts2_FM, threshold, threshold2)
    index_xx, group_xx_pts = Find_duplicate_jyz(xx_3d_pts1, xx_3d_pts2, xx_3d_pts1_rep, xx_3d_pts2_rep, T)
    num_xx = group_xx_pts.shape[0]
    return num_xx, group_xx_pts

def Camera_vec_to_world_vec(vec, roll, yaw, pitch):
    '''
    根据云台角度计算导线方向
    '''
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

def Calculate_direction(degree_front):
    # 云台方向向量
    direction_front_3d = Camera_vec_to_world_vec([0, 0, 1], degree_front[0], degree_front[1], degree_front[2])

    # 俯视图下的导线方向向量
    direction_front_2d = direction_front_3d[:2]
    norm = np.linalg.norm(direction_front_2d)

    # 俯视图下的归一化导线方向向量
    direction_front_line_2d = direction_front_2d / norm

    return direction_front_line_2d

def Calculate_distance_jyz_hd(jyz_layer, direction_front_line_2d, xyz_ganta):

    # 横担方向向量
    direction_hengdan = np.array([- direction_front_line_2d[1], direction_front_line_2d[0]])
    jyz_layer2 = jyz_layer.copy()

    # 对每个绝缘子计算其与横担的距离
    for i in range(len(jyz_layer)):

        xy_ganta = xyz_ganta[:2]                # 杆塔xy坐标
        xy_jyz = jyz_layer[i][:2]               # 绝缘子xy坐标
        v = xy_jyz - xy_ganta            # 绝缘子相对于杆塔的向量v

        normal = np.array([-direction_hengdan[1], direction_hengdan[0]])

        # 绝缘子方向
        direction = np.dot(v, direction_front_line_2d)
        if direction >= 0:
            direction = 1
        else:
            direction = -1
        distance = direction * np.abs(np.dot(v, normal)) / np.linalg.norm(normal)

        jyz_layer2[i] = np.hstack((jyz_layer2[i], distance))

    return jyz_layer2

def Calculate_jyz_direction(jyz_layer2):

    # 提取所有的距离值
    values = [row[9] for row in jyz_layer2]

    # 找到最大的和第二大的值的索引
    max_index = np.argmax(values)
    values[max_index] = -np.inf  # 将最大值设为负无穷，以便找到第二大的值
    second_max_index = np.argmax(values)

    # 恢复最大值的原始值
    values[max_index] = [row[6] for row in jyz_layer2][max_index]

    jyz_direction = jyz_layer2[max_index][:2] - jyz_layer2[second_max_index][:2]
    return jyz_direction

def Calculate_delta_degree(direction_jyz, direction_hengdan):

    # 计算点积
    dot_product = np.dot(direction_jyz, direction_hengdan)

    # 计算模
    magnitude1 = np.linalg.norm(direction_jyz)
    magnitude2 = np.linalg.norm(direction_hengdan)

    # 计算余弦值
    cos_theta = dot_product / (magnitude1 * magnitude2)

    # 计算角度（弧度）
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # 转换为度数
    delta_degree = np.degrees(theta_rad)

    # 确保角度是锐角
    if delta_degree > 90:
        delta_degree = 180 - delta_degree

    return delta_degree

def Calculate_jyz_on_line(jyz_layer2, direction_jyz):
    """
    根据绝缘子直线计算哪些绝缘子分布在绝缘子直线上
    """
    jyz_layer_on_line = []

    # 提取所有的距离值
    values = [row[9] for row in jyz_layer2]

    # 找到最大的和第二大的值的索引
    max_index = np.argmax(values)

    # 绝缘子直线必定通过的点A
    A = jyz_layer2[max_index][:2]

    for i in range(len(jyz_layer2)):

        jyz_2d_pt = jyz_layer2[i][:2]

        # 计算直线L1的法向量
        normal = np.array([-direction_jyz[1], direction_jyz[0]])

        # 计算点A到jyz的向量
        AB = jyz_2d_pt - A

        # jyz到绝缘子直线的距离
        distance = np.abs(np.dot(AB, normal)) / np.linalg.norm(normal)

        if distance < 0.2:

            jyz_layer_on_line.append(jyz_layer2[i])

    return jyz_layer_on_line

def sort_points_by_direction(points, direction):
    """
    根据绝缘子向量对绝缘子进行排序。
    """
    # 将方向向量归一化
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)

    # 计算每个点与方向向量的点积
    def projection(point):
        return np.dot(np.array(point[:2]), direction)

    # 对点进行排序
    sorted_points = sorted(points, key=projection)

    return sorted_points

