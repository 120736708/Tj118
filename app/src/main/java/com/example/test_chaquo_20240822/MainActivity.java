package com.example.test_chaquo_20240822;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.app.AlertDialog;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;


import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    public class Obj
    {
        public float x;
        public float y;
        public float w;
        public float h;
        public String label;
        public float prob;
    }
    Button Go_btn, button;
    ImageView src_image, res_image;
    BitmapDrawable drawable;
    Bitmap bitmap;
    // 定义一个请求代码，用于识别权限请求的结果
    private static final int MY_PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE = 123; // 123 是一个示例值，可以是任何唯一的整数

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Go_btn = findViewById(R.id.Go_button);
        button = findViewById(R.id.button);
        ImageView src_image = findViewById(R.id.source_imageview);
        ImageView res_image = findViewById(R.id.response_imageview);
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
        Python python_cv = Python.getInstance();

        JSONObject json1 = new JSONObject();

        try {
            // 构建startLoc对象
            JSONObject startLoc = new JSONObject();
            startLoc.put("lat", 30.12423525);
            startLoc.put("lon", 114.142525);
            startLoc.put("alt", 30.4);

            // 构建towerLoc对象
            JSONObject towerLoc = new JSONObject();
            towerLoc.put("lat", 30.12423525);
            towerLoc.put("lon", 114.142525);
            towerLoc.put("alt", 30.4);

            // 构建objs数组中的单个对象
            JSONObject obj = new JSONObject();
            obj.put("x", 123);
            obj.put("y", 345);
            obj.put("w", 23);
            obj.put("h", 333);
            obj.put("label", "1233");
            obj.put("prob", 99.4);

            // 构建objs数组
            JSONArray objs = new JSONArray();
            objs.put(obj);

            // 构建photoData数组中的单个对象
            JSONObject photoDataItem = new JSONObject();
            photoDataItem.put("data", 123121414);
            photoDataItem.put("location", startLoc); // 使用startLoc作为示例位置
            photoDataItem.put("roll", 30.12);
            photoDataItem.put("yaw", 23.12);
            photoDataItem.put("pitch", 12.22);
            photoDataItem.put("objs", objs);

            // 构建photoData数组
            JSONArray photoData = new JSONArray();
            photoData.put(photoDataItem);

            // 将startLoc, towerLoc和photoData添加到json1中
            json1.put("startLoc", startLoc);
            json1.put("towerLoc", towerLoc);
            json1.put("photoData", photoData);

            // 打印json1字符串，用于验证
            Log.d("JSON", json1.toString());

        } catch (Exception e) {
            e.printStackTrace();
        }

        Go_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                String workspace = Environment.getExternalStorageDirectory().getPath() + "/etower";
                Log.d("WorkspaceTag", "Workspace path: " + workspace);
                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                // 获取源图片并转换为Bitmap对象
                drawable = (BitmapDrawable) getDrawable(R.drawable.test0);
                bitmap = drawable.getBitmap();

                // 该段代码耗时较长，将Bitmap对象压缩成jpg格式
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
                byte[] byteArray0 = stream.toByteArray();

                drawable = (BitmapDrawable) getDrawable(R.drawable.test1);
                bitmap = drawable.getBitmap();
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
                byte[] byteArray1 = stream.toByteArray();

                drawable = (BitmapDrawable) getDrawable(R.drawable.test2);
                bitmap = drawable.getBitmap();
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
                byte[] byteArray2 = stream.toByteArray();

                // 函数输入
                double[] loc0 = {30.3067, 114.4250, 38.7490};
                double[] loc_ganta = {30.3067, 114.4250, 38.7490};
                double[] loc_mov1 = {30.3067055, 114.4250684, 38.777};
                double[] loc_ref = {30.3067068, 114.4250727, 38.771};
                double[] loc_mov2 = {30.3067065, 114.4250782, 38.755};
                double[] degree_mov1 ={0, -2.3, -54.4};
                double[] degree_ref = {0, -5.1, -54.4};
                double[] degree_mov2 = {0, -11, -54.4};

                List<Obj> box_obj_mov1 = new ArrayList<>();
                Obj obj = new Obj();
                obj.x = 1773;
                obj.y = 1546;
                obj.h = 50;
                obj.w = 50;
                obj.label = "jueyuanzi";
                box_obj_mov1.add(obj);
                Obj obj1 = new Obj();
                obj1.x = 1673;
                obj1.y = 1246;
                obj1.h = 50;
                obj1.w = 50;
                obj1.label = "hengdan";
                box_obj_mov1.add(obj1);
                String[][] result0 = new String[box_obj_mov1.size()][5];

                // 填充二维数组
                for (int i = 0; i < box_obj_mov1.size(); i++) {
                    Obj obj0 = box_obj_mov1.get(i);
                    result0[i][0] = String.valueOf(obj0.x);      // x
                    result0[i][1] = String.valueOf(obj0.y);      // y
                    result0[i][2] = String.valueOf(obj0.w);      // w
                    result0[i][3] = String.valueOf(obj0.h);      // h
                    result0[i][4] = obj0.label;                  // label
                }

                List<Obj> box_obj_ref = new ArrayList<>();
                obj.x = 1773;
                obj.y = 1546;
                obj.h = 50;
                obj.w = 50;
                obj.label = "jueyuanzi";
                box_obj_ref.add(obj);

                obj.x = 1773;
                obj.y = 1546;
                obj.h = 50;
                obj.w = 50;
                obj.label = "hengdan";
                box_obj_ref.add(obj);
                String[][] result1 = new String[box_obj_ref.size()][5];

                // 填充二维数组
                for (int i = 0; i < box_obj_ref.size(); i++) {
                    Obj obj0 = box_obj_ref.get(i);
                    Log.d("WorkspaceTag", "obj0: " + obj0);
                    result1[i][0] = String.valueOf(obj0.x);      // x
                    result1[i][1] = String.valueOf(obj0.y);      // y
                    result1[i][2] = String.valueOf(obj0.w);      // w
                    result1[i][3] = String.valueOf(obj0.h);      // h
                    result1[i][4] = obj.label;                  // label
                }


                List<Obj> box_obj_mov2 = new ArrayList<>();
                obj.x = 1773;
                obj.y = 1546;
                obj.h = 50;
                obj.w = 50;
                obj.label = "jueyuanzi";
                box_obj_mov2.add(obj);
                obj.x = 1773;
                obj.y = 1546;
                obj.h = 50;
                obj.w = 50;
                obj.label = "hengdan";
                box_obj_mov2.add(obj);
                String[][] result2 = new String[box_obj_mov2.size()][5];

                // 填充二维数组
                for (int i = 0; i < box_obj_mov2.size(); i++) {
                    Obj obj0 = box_obj_mov2.get(i);
                    result2[i][0] = String.valueOf(obj0.x);      // x
                    result2[i][1] = String.valueOf(obj0.y);      // y
                    result2[i][2] = String.valueOf(obj0.w);      // w
                    result2[i][3] = String.valueOf(obj0.h);      // h
                    result2[i][4] = obj.label;                  // label
                }




                // 调用Python方法计算物料空间点
                PyObject cvObject = python_cv.getModule("Image2Cor");
//                PyObject res = cvObject.callAttr("image2Cor", loc0, loc_ganta, loc_mov1, loc_ref, loc_mov2,
//                        degree_mov1, degree_ref, degree_mov2, byteArray0, byteArray1, byteArray2,
//                        result0, result1, result2);
                String jsonString = json1.toString();
                PyObject res = cvObject.callAttr("image2Cor", jsonString);
                double[] degree_front = {0, 161.3, -54.4};
                double[] degree_back = {0, 32.1, -54.4};
                // 调用Python方法计算空间测距跟数量统计
//                PyObject cv1bject = python_cv.getModule("Cor2Result");
//                PyObject res = cv1bject.callAttr("cor2Result", Result1, Result2, loc0, loc_ganta, degree_front, degree_back);
//                byte[] bytes = cvObject.callAttr("opencv_process_image", byteArray).toJava(byte[].class);
//                // 将处理后的图片显示到画面上
//                Bitmap bmp = BitmapFactory.decodeByteArray(bytes,0, bytes.length);
//                res_image.setImageBitmap(bmp);


                //中间弹出信息
                AlertDialog textTips = new AlertDialog.Builder(MainActivity.this)
                        .setTitle("Tips:")
                        .setMessage("" + res)
                        .create();
                textTips.show();
            }
        });
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                requestPermission();
            }
        });
    }
    private static final int REQUEST_MANAGE_FILES_ACCESS = 2;

    public void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            //判断是否有管理外部存储的权限
            if (!Environment.isExternalStorageManager()) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                intent.setData(Uri.parse("package:" + getPackageName()));
                startActivityForResult(intent, REQUEST_MANAGE_FILES_ACCESS);
            } else {
                // 已有所有文件访问权限，可直接执行文件相关操作
                Toast.makeText(this, "已获取所有文件访问权限", Toast.LENGTH_LONG).show();
            }
        } else {
            //非android11及以上版本，走正常申请权限流程
            Toast.makeText(this, "非Android11， 默认授权成功", Toast.LENGTH_LONG).show();
        }
    }

}



