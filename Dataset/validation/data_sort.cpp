#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>

using namespace std;

int main() {
  vector<string> name = {"03.12.2019-14.43.51_","03.12.2019-14.43.52_","03.12.2019-14.43.53_","05.12.2019-11.28.50_"};
  vector<int> size = {3122,5945,43593,27140};
  ofstream fout;
  ifstream fin;
  string img, cmd;
  int lb, index;
  double imu1, imu2, imu3, imu4, imu5, imu6;
  vector <string> img_vec;
  vector <int> lb_vec;
  vector <double> imu1_vec, imu2_vec, imu3_vec, imu4_vec, imu5_vec, imu6_vec;
  bool flag;
/*
03.12.2019-14.43.51_41     03.12.2019-14.43.52_5     03.12.2019-14.43.53_65     05.12.2019-11.28.50_60
03.12.2019-14.43.51_3122   03.12.2019-14.43.52_5945  03.12.2019-14.43.53_43593  05.12.2019-11.28.50_27140
*/

  fin.open("data.txt",ios::in);
  while( !fin.eof() ) {
    fin >> img >> lb >> imu1 >> imu2 >> imu3 >> imu4 >> imu5 >> imu6;
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
    if( fin.eof() )
      break;
  }
  fin.close();

  fout.open( "all_data_sort.txt", ios::out );
  for (int i = 0; i < name.size(); i++ ) {
    for( int j = 0; j <= size[i]; j++ ) {
      //cout << name[ i ] + to_string( j ) + ".jpg" << endl;
      // CHECK IF name[i] + to_string(j) + ".jpg" exist
      flag = false;
      for( int k = 0; k < img_vec.size(); k++ ) {
        cmd = "non_obs/" + name[ i ] + to_string( j ) + ".jpg";
        if( cmd ==  img_vec[ k ] ) {
          flag = true;
          index = k;
          break;
        }
        cmd = "obs/" + name[ i ] + to_string( j ) + ".jpg";
        if( cmd ==  img_vec[ k ] ) {
          flag = true;
          index = k;
          break;
        }
      }
      if( flag ) {
        fout << "all/" + name[ i ] + to_string( j ) + ".jpg ";
        fout << " 0 " << setprecision( 8 );
        fout << imu1_vec[ index ] << " " << imu2_vec[ index ] << " ";
        fout << imu3_vec[ index ] << " " << imu4_vec[ index ] << " ";
        fout << imu5_vec[ index ] << " " << imu6_vec[ index ] <<  endl;
      }
    }
  }
  fout.close();


  return 0;

}
