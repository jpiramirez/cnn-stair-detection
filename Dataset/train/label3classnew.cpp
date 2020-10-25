#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>

using namespace std;

int main( ) {
  ifstream f,f2;
  ofstream f3;
  string cmd, start_img, end_img;
  string prefix="all/";
  vector<string> img_vec, simg_vec, eimg_vec;
  string img;
  vector<int> lb_vec, selb_vec;
  int lb;
  double imu1, imu2, imu3, imu4, imu5, imu6;
  vector<double> imu1_vec, imu2_vec, imu3_vec, imu4_vec, imu5_vec, imu6_vec;

  f.open( "all_data_sort.txt", ios::in );
  f2.open("label_3class_new.txt", ios::in );
  f2 >> cmd >> cmd >> cmd >> cmd;

  // Load all data from f
  while( !f.eof() ) {
    f >> img >> lb >> imu1 >> imu2 >> imu3 >> imu4 >> imu5 >> imu6;
    //cout << img << " " << lb << " " << imu1 << " " << imu2 << " ";
    //cout << imu3 << " " << imu4 << " " << imu5 << " " << imu6 <<  endl;
    img_vec.push_back( img );
    lb_vec.push_back( lb );
    imu1_vec.push_back( imu1 );
    imu2_vec.push_back( imu2 );
    imu3_vec.push_back( imu3 );
    imu4_vec.push_back( imu4 );
    imu5_vec.push_back( imu5 );
    imu6_vec.push_back( imu6 );
    if( f.eof() )
      break;
  }
  f.close();
  //Load all data from f2
  while( !f2.eof() ) {
    f2 >> start_img >> end_img >> lb;
    //cout << start_img << " " << end_img << " " << lb << "\n";
    simg_vec.push_back( start_img );
    eimg_vec.push_back( end_img );
    selb_vec.push_back( lb );
    if( f2.eof() )
      break;
  }
  f2.close();

  bool flag = false;
  f3.open("data_3class_new.txt", ios::out );
  for( int i = 0; i < img_vec.size(); i++ ) {
    // Search in simg
    flag = false;
    for( int j = 0; j < simg_vec.size(); j++ ) {
      if( prefix + simg_vec[ j ] == img_vec[ i ] ) {
        flag = true;
        // incrementar i hasta que sea igual a eimg_vec[ j ]
        while( prefix + eimg_vec[ j ] != img_vec[ i ] ) {
          f3 << img_vec[ i ] << " " << selb_vec[ j ] << " " << setprecision( 8 ) << imu1_vec[ i ] << " " << imu2_vec[ i ] << " ";
          f3 << imu3_vec[ i ] << " " << imu4_vec[ i ] << " " << imu5_vec[  i ] << " ";
          f3 << imu6_vec[ i ] <<  endl;
          i++;
        }
        f3 << img_vec[ i ] << " " << selb_vec[ j ] << " " << setprecision( 8 ) << imu1_vec[ i ] << " " << imu2_vec[ i ] << " ";
        f3 << imu3_vec[ i ] << " " << imu4_vec[ i ] << " " << imu5_vec[  i ] << " ";
        f3 << imu6_vec[ i ] <<  endl;
        break;
      }
    }
    // Vacia los datos de i con etiqueta 0
    if( !flag ) {
      f3 << img_vec[ i ] << " 0 " << setprecision( 8 ) << imu1_vec[ i ] << " " << imu2_vec[ i ] << " ";
      f3 << imu3_vec[ i ] << " " << imu4_vec[ i ] << " " << imu5_vec[  i ] << " ";
      f3 << imu6_vec[ i ] <<  endl;
    }

  }
  f3.close();

  return 0;
}
