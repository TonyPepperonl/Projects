#include <cmath>     // exp, ceil
#include <cstdio>    // printf
#include <cstdlib>   // atof, atoi
#include <vector>    // std::vector
#include <omp.h>     // omp_get_wtime, omp_get_max_threads
#include <string>
#include <algorithm> // std::min, std::max

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

double count = -1.0;
double total_px = 0;

int progress(){
    count = count + 1.0;
    printf("\rPercent complete:  |%.*s%.*s|  %.2f%%", int(((count/total_px)*100)/10), "**********",int(10-((count/total_px)*100)/10), "----------", (count/total_px)*100);
    return 0;
}

int main(int argc, char** argv) {
    const char* input = "test_image.jpg";
    const char* output = "test_output.jpg";
    if (argc < 3) {
        std::printf("\n*Missing Arguments*\nusage: %s [input.jpg] [output.jpg] [sigma]\n\n**Test data will be used for this attempt.\n\n", argv[0]);
    }else if (argc == 2){
        input =argv[1];
    }
    else{
        input =argv[1];
        output =argv[2];
    }
    bool devlog = false;
    if(argc >= 5 && std::string(argv[4]) == "--dev") {
        devlog = true;
        
    }

    


    double sigma = (argc >= 4) ? std::atof(argv[3]) : 5.0; 
    if (sigma <= 0.0) sigma = 1.0;

    int w, h, channels;
    unsigned char* img = stbi_load(input, &w, &h, &channels, 0);
    if (!img) {
        std::printf("could not load %s\n", input);
        return 1;
    }

    int radius = (int)std::ceil(3.0 * sigma); 
    int ksize  = 2 * radius + 1; 
    std::vector<double> kernel(ksize); 

    double sum = 0.0;
    for (int d = -radius; d <= radius; ++d) {
        double val = std::exp(-(d * d) / (2.0 * sigma * sigma));
        kernel[d + radius] = val;
        sum += val;
        }
    
    for (double& val: kernel) val /= sum;
    
    
    size_t img_bytes = static_cast<size_t>(w) * h * channels;
    std::vector<double> temp_buffer(img_bytes);
    std::vector<unsigned char> out(static_cast<size_t>(w) * h * channels); 

    int rows = h;

    double t0 = omp_get_wtime();
    if(devlog) {
        printf("echo -e \"\\e[?25h\" <- to get cursor back if program terminates\n");
        printf("Starting..");
        total_px = double(h);
        progress();
        fputs("\e[?25l", stdout); 
    }

    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t pixel_idx = (static_cast<size_t>(y) * w + x) * channels;
            
            for (int c = 0; c < channels; ++c) {
                double acc = 0.0;
                for (int dx = -radius; dx <= radius; ++dx) {
                    int xx = std::max(0, std::min(w - 1, x + dx));
                    size_t neighbor_idx = (static_cast<size_t>(y) * w + xx) * channels + c;
                    acc += kernel[dx + radius] * img[neighbor_idx];
                }
                temp_buffer[pixel_idx + c] = acc;    
            }
        }
        if (devlog) {
            #pragma omp critical
            progress();
        }
    }

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t pixel_idx = (static_cast<size_t>(y) * w + x) * channels;
            
            for (int c = 0; c < channels; ++c) {
                double acc = 0.0;
                for (int dy = -radius; dy <= radius; ++dy) {
                    int yy = std::max(0, std::min(h - 1, y + dy));
                    size_t neighbor_idx = (static_cast<size_t>(yy) * w + x) * channels + c;
                    acc += kernel[dy + radius] * temp_buffer[neighbor_idx];
                }
                out[pixel_idx + c] = static_cast<unsigned char>(acc + 0.5);
            }
        }
        if (devlog) {
            #pragma omp critical
            progress();
        }
    }
    
    if (devlog) {
        fputs("\e[?25h", stdout);
        std::printf("\nWriting Image to jpg...\n");
    }

double t1 = omp_get_wtime();

stbi_write_jpg(output, w, h, channels, out.data(), 90);
stbi_image_free(img);

std::printf("blurred %dx%d (%d ch), sigma=%.2f, %d threads, %.3f s\n", w, h, channels, sigma, omp_get_max_threads(), t1 - t0);
return 0;
}