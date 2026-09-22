#include <iostream>
#include <iomanip>
#include <cmath>
#include <array>
#include <vector>
#include <queue>
#include <limits>
#include <stdexcept>
#include <sstream>

// Nós positivos da regra Kronrod de 21 pontos, em ordem decrescente.
inline constexpr std::array<double, 11> XGK = {
    0.9956571630258081,
    0.9739065285171717,
    0.9301574913557082,
    0.8650633666889845,
    0.7808177265864169,
    0.6794095682990244,
    0.5627571346686047,
    0.4333953941292472,
    0.2943928627014602,
    0.1488743389816312,
    0.0
};

// Pesos da regra Gauss de 10 pontos. Os seus nós são XGK[1], [3], ..., [9].
inline constexpr std::array<double, 5> WG = {
    0.06667134430868814,
    0.1494513491505806,
    0.21908636251598204,
    0.26926671930999635,
    0.29552422471475287
};

// Pesos correspondentes aos nós em XGK.
inline constexpr std::array<double, 11> WGK = {
    0.011694638867371874,
    0.03255816230796473,
    0.054755896574351996,
    0.07503967481091995,
    0.09312545458369761,
    0.10938715880229764,
    0.12349197626206585,
    0.13470921731147333,
    0.14277593857706008,
    0.14773910490133849,
    0.1494455540029169
};

struct ResultadoQuad {
    double valor;
    double erro;
};

template <typename Func>
ResultadoQuad gauss_kronrod_21(Func&& funcao, double inicio, double fim) {
    double centro = (inicio + fim) / 2.0;
    double semi_largura = (fim - inicio) / 2.0;

    double valor_central = funcao(centro);
    double soma_gauss = 0.0;
    double soma_kronrod = WGK[10] * valor_central;

    struct ValorPeso {
        double valor;
        double peso;
    };
    std::array<ValorPeso, 21> valores;
    valores[0] = {valor_central, WGK[10]};
    std::size_t num_valores = 1;

    for (std::size_t indice = 0; indice < 10; ++indice) {
        double no = XGK[indice];
        double deslocamento = semi_largura * no;
        double valor_esquerda = funcao(centro - deslocamento);
        double valor_direita = funcao(centro + deslocamento);
        double soma_pares = valor_esquerda + valor_direita;

        soma_kronrod += WGK[indice] * soma_pares;
        valores[num_valores++] = {valor_esquerda, WGK[indice]};
        valores[num_valores++] = {valor_direita, WGK[indice]};

        // Os nós de Gauss estão nas posições ímpares da tabela de Kronrod.
        if (indice % 2 == 1) {
            soma_gauss += WG[(indice - 1) / 2] * soma_pares;
        }
    }

    double integral_kronrod = semi_largura * soma_kronrod;
    double integral_gauss = semi_largura * soma_gauss;
    double erro = std::abs(integral_kronrod - integral_gauss);

    // Ajuste do QUADPACK: a variação das avaliações face à média permite
    // evitar uma estimativa de erro demasiado otimista.
    double media = soma_kronrod / 2.0;
    double soma_variacao = 0.0;
    double soma_abs = 0.0;
    for (const auto& vp : valores) {
        soma_variacao += vp.peso * std::abs(vp.valor - media);
        soma_abs += vp.peso * std::abs(vp.valor);
    }

    double variacao = std::abs(semi_largura) * soma_variacao;
    if (variacao != 0.0 && erro != 0.0) {
        erro = variacao * std::min(1.0, std::pow(200.0 * erro / variacao, 1.5));
    }

    double erro_minimo = 50.0 * std::numeric_limits<double>::epsilon() * std::abs(semi_largura) * soma_abs;
    return {integral_kronrod, std::max(erro, erro_minimo)};
}

struct Subintervalo {
    double esquerda;
    double direita;
    double valor;
    double erro;

    // Max-heap por defeito em std::priority_queue
    bool operator<(const Subintervalo& outro) const noexcept {
        return erro < outro.erro;
    }
};

template <typename Func>
ResultadoQuad quad(
    Func&& funcao,
    double inicio,
    double fim,
    double epsabs = 1e-8,
    double epsrel = 1e-8,
    std::size_t limite = 1000
) {
    if (epsabs < 0.0 || epsrel < 0.0) {
        throw std::invalid_argument("epsabs e epsrel devem ser não negativos");
    }
    if (limite < 1) {
        throw std::invalid_argument("limite deve ser pelo menos 1");
    }
    if (inicio == fim) {
        return {0.0, 0.0};
    }
    if (inicio > fim) {
        auto res = quad(funcao, fim, inicio, epsabs, epsrel, limite);
        return {-res.valor, res.erro};
    }

    auto [valor, erro] = gauss_kronrod_21(funcao, inicio, fim);
    std::priority_queue<Subintervalo> subintervalos;
    subintervalos.push({inicio, fim, valor, erro});

    double integral_total = valor;
    double erro_total = erro;

    while (erro_total > std::max(epsabs, epsrel * std::abs(integral_total))) {
        if (subintervalos.size() >= limite) {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(3) << erro_total;
            throw std::runtime_error(
                "não foi atingida a tolerância após " + std::to_string(limite) +
                " subintervalos; erro estimado: " + oss.str()
            );
        }

        Subintervalo topo = subintervalos.top();
        subintervalos.pop();

        double meio = (topo.esquerda + topo.direita) / 2.0;
        auto [valor_esq, erro_esq] = gauss_kronrod_21(funcao, topo.esquerda, meio);
        auto [valor_dir, erro_dir] = gauss_kronrod_21(funcao, meio, topo.direita);

        integral_total += valor_esq + valor_dir - topo.valor;
        erro_total += erro_esq + erro_dir - topo.erro;

        subintervalos.push({topo.esquerda, meio, valor_esq, erro_esq});
        subintervalos.push({meio, topo.direita, valor_dir, erro_dir});
    }

    return {integral_total, erro_total};
}

double f(double x) {
    return std::pow(x, 2.0) + 2.0 * x + 4.0;
}

int main() {
    auto [valor, erro] = quad(f, 0.0, 3.0);
    std::cout << std::fixed << std::setprecision(15);
    std::cout << "integral = " << valor << "\n";
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "erro estimado = " << erro << "\n";

    return 0;
}
