// ============================================================
// NON-IID DISTRIBUTION HELPERS
// Use compile-time flags:
//
// -DROUND_ROBIN_BASELINE
// -DLABEL_SHARD_NONIID
// -DROTATE_FEATURE_SKEW
//
// Example:
//
// mpicxx main.cpp -DROUND_ROBIN_BASELINE
//
// mpicxx main.cpp -DLABEL_SHARD_NONIID -DROTATE_FEATURE_SKEW
//
// ============================================================

#include <cmath>
#include <vector>
// ------------------------------------------------------------
// 1. IID BASELINE : ROUND ROBIN DISTRIBUTION
//
// Client 1 gets sample 0, N, 2N...
// Client 2 gets sample 1, N+1...
//
// Strong IID baseline
// ------------------------------------------------------------

static std::vector<Image> round_robin_distribution(
    const std::vector<Image>& dataset,
    int data_holder_num,
    int num_data_holders)
{
    std::vector<Image> local_data;

    // rank 1 -> index 0
    int target_index = data_holder_num - 1;

    for (int i = 0; i < (int)dataset.size(); i++) {
        if (i % num_data_holders == target_index) {
            local_data.push_back(dataset[i]);
        }
    }

    return local_data;
}


// ------------------------------------------------------------
// 2. NON-IID LABEL SKEW
//
// Sort by label then contiguous shard
//
// This is your pathological non-IID FL split
// ------------------------------------------------------------

static std::vector<Image> label_shard_distribution(
    std::vector<Image>& dataset,
    int data_holder_num,
    int num_data_holders)
{
    std::stable_sort(
        dataset.begin(),
        dataset.end(),
        [](const Image &a, const Image &b) {
            return a.label < b.label;
        }
    );

    int data_size = (int)dataset.size();
    int shard_size = data_size / num_data_holders;

    int ibegin = (data_holder_num - 1) * shard_size;
    int iend;

    if (data_holder_num == num_data_holders)
        iend = data_size;
    else
        iend = ibegin + shard_size;

    return std::vector<Image>(
        dataset.begin() + ibegin,
        dataset.begin() + iend
    );
}


// ------------------------------------------------------------
// 3. FEATURE SKEW
//
// Rotate images by:
//
// rank 1 -> 10 degrees
// rank 2 -> 20 degrees
// rank 3 -> 30 degrees
//
// etc.
//
// This simulates domain shift / feature skew
// ------------------------------------------------------------

static std::vector<float> rotate_single_image(
    const std::vector<float>& input,
    int rows,
    int cols,
    float angle_degrees)
{
    std::vector<float> output(rows * cols, 0.0f);

    float angle = angle_degrees * M_PI / 180.0f;
    float cosA = std::cos(angle);
    float sinA = std::sin(angle);

    float cx = (cols - 1) / 2.0f;
    float cy = (rows - 1) / 2.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {

            float tx = x - cx;
            float ty = y - cy;

            float src_x =  cosA * tx + sinA * ty + cx;
            float src_y = -sinA * tx + cosA * ty + cy;

            int sx = (int)std::round(src_x);
            int sy = (int)std::round(src_y);

            if (sx >= 0 && sx < cols &&
                sy >= 0 && sy < rows)
            {
                output[y * cols + x] =
                    input[sy * cols + sx];
            }
        }
    }

    return output;
}


static void apply_rotation_feature_skew(
    std::vector<Image>& local_data,
    int data_holder_num)
{
    float angle = 10.0f * data_holder_num;

    for (auto& img : local_data) {
        img.image = rotate_single_image(
            img.image,
            28,
            28,
            angle
        );
    }

    std::cout
        << "[Worker " << data_holder_num
        << "] Applied feature skew rotation: "
        << angle << " degrees\n";
}
