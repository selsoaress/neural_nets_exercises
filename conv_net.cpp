#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

// Usamos namespace para simplificar, mas em produção é melhor evitar.
using namespace std; 

// --- 1. CLASSE BASE: TENSOR 3D ---
template <typename T>
class Tensor {
private:
    vector<T> data_;
    size_t rows_;
    size_t cols_;
    size_t depth_;

public:
    Tensor(size_t rows, size_t cols, size_t depth) 
    : rows_(rows), cols_(cols), depth_(depth), data_(rows * cols * depth) {
        // Inicializa dados com zeros
        fill(data_.begin(), data_.end(), 0.0);
    } 

    // Acesso indexado correto (i, j, k) - Row-Major
    T& operator()(size_t i, size_t j, size_t k) {
        return data_[i * cols_ * depth_ + j * depth_ + k];
    }

    const T& operator()(size_t i, size_t j, size_t k) const { 
        return data_[i * cols_ * depth_ + j * depth_ + k];
    }
    
    // Métodos utilitários
    size_t getRows() const { return rows_; }
    size_t getCols() const { return cols_; }
    size_t getDepth() const { return depth_; }
    size_t getTotalSize() const { return data_.size(); }
};

// --- 2. CLASSE: CAMADA CONVOLUCIONAL (CONV) ---
template <typename T>
class ConvLayer {
private:
    vector<Tensor<T>> filters_; // Pesos (W)
    vector<T> bias_;             // Bias (b)
    
    size_t kernelSize_;
    size_t stride_;
    size_t padding_;

public:
    ConvLayer(size_t numFilters, size_t kernelSize, size_t inputDepth, size_t stride = 1, size_t padding = 0)
        : kernelSize_(kernelSize), stride_(stride), padding_(padding) {
        
        // Inicializa filtros e bias
        for (size_t i = 0; i < numFilters; ++i) {
            // Cada filtro: KxKxInputDepth
            filters_.emplace_back(kernelSize, kernelSize, inputDepth); 
            // TODO: Adicionar inicialização aleatória dos pesos (ex: He/Xavier)
        }
        bias_.resize(numFilters, 0.0);
    }

    Tensor<T> forward(const Tensor<T>& input) {
        size_t inputRows = input.getRows();
        size_t inputCols = input.getCols();
        size_t numFilters = filters_.size();

        // Cálculo das dimensões de saída (Hout, Wout)
        size_t outputRows = (inputRows - kernelSize_ + 2 * padding_) / stride_ + 1;
        size_t outputCols = (inputCols - kernelSize_ + 2 * padding_) / stride_ + 1;
        
        // Cria o tensor de saída (OutputRows x OutputCols x NumFilters)
        Tensor<T> output(outputRows, outputCols, numFilters);

        // Loop sobre cada filtro (profundidade da saída)
        for (size_t f = 0; f < numFilters; ++f) {
            const Tensor<T>& filter = filters_[f];
            
            // Loop sobre as posições de saída (i, j)
            for (size_t i = 0; i < outputRows; ++i) {
                for (size_t j = 0; j < outputCols; ++j) {
                    
                    // Posição inicial do kernel na entrada
                    size_t startRow = i * stride_ - padding_;
                    size_t startCol = j * stride_ - padding_;
                    
                    T activation = 0.0;
                    
                    // Loop sobre o kernel (Kh, Kw, Kin)
                    for (size_t kr = 0; kr < kernelSize_; ++kr) {
                        for (size_t kc = 0; kc < kernelSize_; ++kc) {
                            for (size_t kd = 0; kd < input.getDepth(); ++kd) {
                                size_t currentR = startRow + kr;
                                size_t currentC = startCol + kc;
                                
                                // Verifica limites (lógica do padding zero)
                                if (currentR >= 0 && currentR < inputRows && 
                                    currentC >= 0 && currentC < inputCols) {
                                    
                                    // SOMA DOS PRODUTOS (Convolução)
                                    activation += input(currentR, currentC, kd) * filter(kr, kc, kd);
                                }
                            }
                        }
                    }
                    
                    // Adiciona o Bias
                    activation += bias_[f];
                    
                    // Armazena no mapa de características
                    output(i, j, f) = activation;
                }
            }
        }
        return output;
    }
};

// --- 3. CLASSE: CAMADA DE ATIVAÇÃO (ReLU) ---
template <typename T>
class ReLULayer {
public:
    Tensor<T> forward(const Tensor<T>& input) {
        Tensor<T> output(input.getRows(), input.getCols(), input.getDepth());
        size_t totalSize = input.getTotalSize();

        for (size_t i = 0; i < totalSize; ++i) {
            // ReLU: f(x) = max(0, x)
            output.data_[i] = std::max((T)0.0, input.data_[i]);
        }
        return output;
    }
};

// --- 4. CLASSE: CAMADA DE POOLING (MAX POOLING) ---
template <typename T>
class MaxPoolingLayer {
private:
    size_t poolSize_;
    size_t stride_;

public:
    MaxPoolingLayer(size_t poolSize = 2, size_t stride = 2) 
        : poolSize_(poolSize), stride_(stride) {}

    Tensor<T> forward(const Tensor<T>& input) {
        size_t inputRows = input.getRows();
        size_t inputCols = input.getCols();
        size_t depth = input.getDepth();
        
        // Cálculo das dimensões de saída
        size_t outputRows = (inputRows - poolSize_) / stride_ + 1;
        size_t outputCols = (inputCols - poolSize_) / stride_ + 1;
        
        Tensor<T> output(outputRows, outputCols, depth);
        
        // Loop sobre a profundidade e as posições de saída
        for (size_t d = 0; d < depth; ++d) {
            for (size_t i = 0; i < outputRows; ++i) {
                for (size_t j = 0; j < outputCols; ++j) {
                    
                    size_t startRow = i * stride_;
                    size_t startCol = j * stride_;
                    
                    T maxValue = -std::numeric_limits<T>::max(); // Inicializa com valor mínimo
                    
                    // Loop sobre a janela de pooling
                    for (size_t pr = 0; pr < poolSize_; ++pr) {
                        for (size_t pc = 0; pc < poolSize_; ++pc) {
                            
                            // MAX POOLING: Encontra o valor máximo na janela
                            maxValue = std::max(maxValue, 
                                                input(startRow + pr, startCol + pc, d));
                        }
                    }
                    output(i, j, d) = maxValue;
                }
            }
        }
        return output;
    }
};

// --- 5. CLASSE: MODELO CNN (A Arquitetura) ---
template <typename T>
class SimpleCNN {
private:
    ConvLayer<T> conv1_;
    ReLULayer<T> relu1_;
    MaxPoolingLayer<T> pool1_;
    // ... Aqui iriam as camadas FC (Fully Connected) para classificação

public:
    SimpleCNN() 
        // CONV1: 16 filtros, tamanho 3x3, entrada 3 canais (RGB), stride 1, padding 0
        : conv1_(16, 3, 3, 1, 0), 
          relu1_(), 
          pool1_(2, 2) {} // MAX POOL: janela 2x2, stride 2

    Tensor<T> forward(const Tensor<T>& input) {
        Tensor<T> output = conv1_.forward(input);
        output = relu1_.forward(output);
        output = pool1_.forward(output);
        
        // TODO: Adicionar Flattening e Camadas Fully Connected (FC)
        
        return output; 
    }
};

// --- FUNÇÃO PRINCIPAL ---
int main() {
    // 1. Cria a Imagem de Entrada (ex: 32x32x3)
    // O ideal seria carregar uma imagem real aqui.
    Tensor<double> image(32, 32, 3);
    
    // Simula que os pixels têm valores diferentes de zero (para ativação)
    for (size_t i = 0; i < 32; ++i) {
        for (size_t j = 0; j < 32; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                image(i, j, k) = (double)(i * j + k) / 1000.0;
            }
        }
    }

    // 2. Cria o Modelo
    SimpleCNN<double> model;

    // 3. Executa o Forward Pass
    Tensor<double> output = model.forward(image);

    // 4. Exibe as Dimensões Finais
    cout << "--- CNN Simples (Forward Pass) ---" << endl;
    cout << "Input: 32x32x3" << endl;
    cout << "CONV1 (Filtros: 16, K: 3x3)" << endl;
    cout << "POOL1 (Max Pool: 2x2)" << endl;
    cout << "Saida Final (Apos Pooling): " 
         << output.getRows() << "x"
         << output.getCols() << "x"
         << output.getDepth() << endl; // Deve ser 15x15x16

    return 0;
}
