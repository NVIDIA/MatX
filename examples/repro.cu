#include <matx.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

auto createMockMnistDatasetMatX(int n = 70000, int d = 784) {
    auto mnist = matx::make_tensor<float>({n, d}, matx::MATX_DEVICE_MEMORY);
    auto mnist_16 = matx::make_tensor<matx::matxFp16>({n, d}, matx::MATX_DEVICE_MEMORY);

    (mnist = matx::random<float>({n, d}, matx::UNIFORM)).run();
    (mnist_16 = matx::as_type<matx::matxFp16>(mnist)).run();

    return mnist_16;
}

auto createMockAMatrixMatX(int n = 70000, int k = 2, int D = 1024) {
    auto A = matx::make_tensor<float>({n, 2*k}, matx::MATX_DEVICE_MEMORY);
    auto A_i = matx::make_tensor<int32_t>({n, 2*k}, matx::MATX_DEVICE_MEMORY);

    int a = 2 * (D - 1);

    (A = matx::random<float>({n, 2*k}, matx::UNIFORM, (float)0, (float)a)).run();
    (A_i = matx::as_type<int32_t>(A)).run();

    return A_i;
}

auto createMockBMatrixMatX(int n = 70000, int m = 2000, int D = 1024) {
    auto B = matx::make_tensor<float>({2*D, m}, matx::MATX_DEVICE_MEMORY);
    auto B_i = matx::make_tensor<int32_t >({2*D, m}, matx::MATX_DEVICE_MEMORY);

    int a = n - 1;

    (B = matx::random<float>({2*D, m}, matx::UNIFORM, (float)0, (float)a)).run();
    (B_i = matx::as_type<int32_t>(B)).run();

    return B_i;
}

int findDistanceBatchSize(float alpha, int n, int d, int k, int m) {
    int batchSize = static_cast<int>((double)(static_cast<long long>(n) * d * 2 * k * m) / (std::pow(1024, 3) * alpha));

    if (batchSize == 0) {
        return n;
    }

    for (int div = batchSize; div > 0; div--) {
        if (n % div == 0) {
            return div;
        }
    }

    return -1; // Should never reach here
}


matx::tensor_t<matx::matxFp16, 2>  findDistancesMatX(matx::tensor_t<matx::matxFp16, 2> &X_t, matx::tensor_t<int32_t, 2> &A_t, matx::tensor_t<int32_t, 2> &B_t, float alpha, int batchSize) {
    const int k = (int)A_t.Shape()[1] / 2;
    const int m = (int)B_t.Shape()[1];

    const int n = (int)X_t.Shape()[0];
    const int d = (int)X_t.Shape()[1];
    int D = (int)B_t.Shape()[0] / 2;

    batchSize = (batchSize != -1) ? batchSize : findDistanceBatchSize(alpha, n, d, k, m);

    auto AFlat_t = matx::flatten(A_t);

    auto distances_t = matx::make_tensor<matx::matxFp16>({n, 2 * k * m}, matx::MATX_DEVICE_MEMORY);

    //int j = 0;
    std::vector<double> times;

    auto start_all = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i += batchSize) {
        auto start = std::chrono::high_resolution_clock::now();

        int maxBatchIdx = i + batchSize - 1; // Index within X along the ROWS

        auto XSubset_t_op = matx::slice(X_t, {i, 0}, {maxBatchIdx + 1, matx::matxEnd});

        auto ABatchFlat_t_op = matx::slice(AFlat_t, {i * 2 * k}, {(maxBatchIdx + 1) * 2 * k});

        auto BBatch_t_op = matx::remap<0>(B_t, ABatchFlat_t_op);

        auto XBatch_t_op = matx::remap<0>(X_t, matx::flatten(BBatch_t_op));

        auto XBatchReshaped_t_op = matx::reshape(XBatch_t_op, {batchSize, 2 * k * m, d});

        auto XSubsetReshaped_t_op = matx::reshape(XSubset_t_op, {batchSize, 1, d});

        auto YBatch_t_op = (XBatchReshaped_t_op - matx::repmat(XSubsetReshaped_t_op, {1, 2 * k * m,
                                                                                      1})); // Repmat is a workaround for minusing naively incompatibhle tensor shapes

        auto YBatch_t_norm_op = matx::vector_norm(YBatch_t_op, {2}, matx::NormOrder::L2);

        (matx::slice(distances_t, {i, 0}, {maxBatchIdx + 1, matx::matxEnd}) = YBatch_t_norm_op).run();

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate the duration
        std::chrono::duration<double> duration = end - start;

        // Cast to double and store in array
        times.push_back(duration.count());
    }

    auto start_sync = std::chrono::high_resolution_clock::now();

    cudaDeviceSynchronize();

    // Record end time
    auto end_sync = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration_sync = end_sync - start_sync;

    // Output the duration
    std::cout << "Sync Time taken: " << duration_sync.count() << " seconds" << std::endl;

    for (const auto &element: times) {
        std::cout << element << std::endl;
    }

    // Record end time
    auto end_all = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end_all - start_all;

    // Output the duration
    std::cout << "Total Time taken: " << duration.count() << " seconds" << std::endl;

    return distances_t;
}

int main() {
int k = 5;
int n = 70000;
int m = 50;
int D = 1024;
int d = 784;

auto A = createMockAMatrixMatX(n, k, D);
auto B = createMockBMatrixMatX(n, m, D);
auto X = createMockMnistDatasetMatX(n, d);

cudaDeviceSynchronize();

auto start = std::chrono::high_resolution_clock::now();

auto distances = findDistancesMatX(X, A, B, 1.2f, 250);
cudaDeviceSynchronize();

// Record end time
auto end = std::chrono::high_resolution_clock::now();

std::chrono::duration<double> duration = end - start;

// Output the duration
std::cout << "Total Time taken (again): " << duration.count() << " seconds" << std::endl;

printf("%lld %lld\n", distances.Shape()[0], distances.Shape()[1]);
return 0;
}
