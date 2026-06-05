# OpenCV 图像处理实验说明
本项目实现了一个基于 OpenCV 的 C++ 图像处理程序，能够读取用户指定的图像文件，输出图像的基本信息（尺寸、通道数、数据类型），并将图像缩放后显示。程序依次完成以下操作：显示原图、转换为灰度图并显示、保存灰度图、裁剪图像左上角 50×50 区域并显示保存、将原图左上角 50×50 区域涂黑后显示保存。所有处理后的图像均保存在程序运行目录下。
## 环境要求
- 操作系统：Linux 或 Windows 10/11 的 WSL 2 环境
- 编译器：支持 C++11 的 g++
- 依赖库：OpenCV 4.x（需安装开发库）
- 图形界面：若在 WSL 中运行，需配置 X 服务器（如 VcXsrv）并设置 DISPLAY 环境变量，以便显示图像窗口
## 使用方法
1. 将源代码保存为 `main.cpp`。
2. 在终端中进入项目目录，执行编译命令（需链接 OpenCV 库）。若使用 pkg-config，命令可简化为：`g++ -g main.cpp -o program $(pkg-config --cflags --libs opencv4)`；若手动指定库，可使用：`g++ -g main.cpp -o program -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs`。
3. 运行生成的可执行文件，根据提示输入图像文件的完整路径。
4. 程序将依次弹出图像窗口，按任意键切换下一张，并在当前目录生成 `gray_image.jpg`、`cropped_region.jpg` 和 `modified_image.jpg` 三个文件。
## 注意事项
- 确保输入的图像路径正确，且文件可读。
- 在 WSL 环境下，需要先启动 X 服务器（如 VcXsrv），并在终端中执行 `export DISPLAY=:0`，否则图像窗口无法显示。
- 如果编译时提示找不到 OpenCV 头文件或链接错误，请检查 OpenCV 是否正确安装，并确认编译命令中库名与安装版本一致。
