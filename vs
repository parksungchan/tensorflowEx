Visual Studio Install
1. Down Visul Studio Community 2015 with Update3https://my.visualstudio.com/Downloads?q=visual%20studio%202015&wt.mc_id=o~msft~vscom~older-downloads
2. Down caffe-windowshttps://github.com/happynear/caffe-windows
3. Down caffe thirdparty (./windows/thirdparty)https://drive.google.com/open?id=0B0OhXbSTAU1HSjM5MUdfZ2RPdFk
4. Make .\windows\CommonSettings.props<CpuOnlyBuild>true</CpuOnlyBuild>    <UseCuDNN>false</UseCuDNN>    <UseNCCL>false</UseNCCL>    <UseMKL>false</UseMKL>    <CudaVersion>8.0</CudaVersion>    <!-- NOTE: If Python support is enabled, PythonDir (below) needs to be         set to the root of your Python installation. If your Python installation         does not contain debug libraries, debug build will not work. -->    <PythonSupport>false</PythonSupport>
    <!-- NOTE: If Matlab support is enabled, MatlabDir (below) needs to be         set to the root of your Matlab installation. -->    <MatlabSupport>false</MatlabSupport>    <MXNetSupport>false</MXNetSupport>   <!-- <CudaDependencies>cufft.lib</CudaDependencies> -->
5. Visul Studio Debug -> Release Change & x64 select
6. library select next properties warnning no change
7. libcaffe build
8. Build -> Build Solution 
9. Release 폴더에 들어가 exe 파일로 실행.
   필요 라이브러리를 설치하고 진행.   Configuration Properties 에서 경로 입력 및 라이브러리 입력을 한다.      
