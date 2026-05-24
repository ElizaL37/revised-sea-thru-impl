// This file is a reimplementation of the illuminant map estimation step 
// in the Sea-Thru algorithm.
//
// This program is free software: you can redistribute it and/or modify  
// it under the terms of the GNU General Public License as published by  
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful, but 
// WITHOUT ANY WARRANTY; without even the implied warranty of 
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License 
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//

#include <queue>
#include <set>
#include <utility>
#include <vector>
#include <iostream>
#include <omp.h>

static int xlim;
static int ylim;


inline int ind(int x, int y) {
    return x * ylim + y;
}


extern "C" {

void compute_illuminant_map(double* Dc, double* depths, double* illu, double p, double f, double eps, int xlim_in, int ylim_in, int iterations) {
    xlim = xlim_in;
    ylim = ylim_in;

    std::vector<double> ac(xlim * ylim, 0.0);
    std::vector<double> ac_new(xlim * ylim, 0.0);

    std::cout << "Computing illuminant." << std::endl;

    // Each pixel av. itself & 4 connected pixel neighbours within eps depth of itself (Sea-Thru eq. 13 & 14)
    for (int k = 0; k < iterations; k++) {
        
        #pragma omp parallel for schedule(static)
    
        for (int x = 0; x < xlim; x++) {
            for (int y = 0; y < ylim; y++) {
                double depth_xy = depths[ind(x,y)];
                double sum = 0.0;
                int count = 0; 
                
                // Pixel Above Pix(x,y)
                if (x > 0 && std::abs(depths[ind(x - 1,y)] - depth_xy) < eps) {
                    sum += ac[ind(x - 1, y)];
                    count++;
                }
                // Pixel Below Pix(x,y)
                if (x < xlim - 1 && std::abs(depths[ind(x + 1,y)] - depth_xy) < eps) {
                    sum += ac[ind(x + 1, y)];
                    count++;
                }
                // Pixel Left of Pix(x,y)
                if (y > 0 && std::abs(depths[ind(x,y - 1)] - depth_xy) < eps) {
                    sum += ac[ind(x, y - 1)];
                    count++;
                }
                // Pixel Right of Pix(x,y)
                if (y < ylim - 1 && std::abs(depths[ind(x,y + 1)] - depth_xy) < eps) {
                    sum += ac[ind(x, y + 1)];
                    count++;
                }

                // If neighbour pixels are within eps, av. their values, else, keep Pix(x,y)'s orig. value
                double ac_p = 0.0;
                if (count > 0) {
                    ac_p = sum / count;
                } else {
                    ac_p = ac[ind(x,y)];
                }
                
                // Eq 14 from Sea-Thru 
                ac_new[ind(x,y)] = Dc[ind(x,y)] * p + ac_p * (1 -p);
            }
        }    
        
        ac.swap(ac_new);
        
    }

    #pragma omp parallel for
    for (int x = 0; x < xlim; x++) {
        for (int y = 0; y < ylim; y++) {
            illu[ind(x, y)] = ac[ind(x, y)] * f;
        }
    }
}

}