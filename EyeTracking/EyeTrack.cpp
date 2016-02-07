#include <iostream>
#include <cstdlib>

using namespace std;
int main()
{
  string curFile = "test1.jpg";
  string cmd = "curl -X POST -d \'\"/home/calvin/testImg/" + curFile + "\"\' -H \'Content-Type: application/json\' -H \'Authorization: Simple simmfOEkamIbcp5G8JSgj4vbQQs1\' https://api.algorithmia.com/v1/algo/opencv/EyeDetection/0.1.1";

  system(cmd.c_str());
  return 0;
}
