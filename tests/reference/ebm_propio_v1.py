"""EBMP: máquina de boosting explicable programada desde cero (modelo original del estudio).

EBMP (Explainable Boosting Machine · Pareja) es un modelo aditivo generalizado para clasificación
binaria,

.. math::

    \\operatorname{logit} \\hat p(x) = \\beta_0 + \\sum_{j=1}^{p} f_j(x_j)
                                   \\; [+ \\sum_{(j,k) \\in \\mathcal{I}} f_{jk}(x_j, x_k)],

en el que cada función de forma :math:`f_j` es escalonada sobre los intervalos (cuantiles) de su
variable y se aprende por **boosting cíclico de Newton**: en cada ronda se recorren las variables una
por una y a cada :math:`f_j` se le suma un pequeño paso que reduce la pérdida logística ponderada.

Lo que lo distingue de un EBM estándar (InterpretML), además de estar escrito solo con numpy:

1. **Paso óptimo por programación dinámica.** El EBM estándar ajusta en cada paso un árbol pequeño
   de forma voraz (greedy). EBMP elige, entre *todas* las particiones de los intervalos ordenados en
   a lo sumo ``max_leaves`` tramos contiguos, la que maximiza la ganancia de Newton
   :math:`\\sum_s G_s^2 / (H_s + \\lambda)`, de manera exacta (ecuación de Bellman, O(L·B²) por paso).
   El valor de cada tramo es el paso de Newton regularizado :math:`-G_s / (H_s + \\lambda)`.
2. **Bolsas vectorizadas.** Las ``n_bags`` bolsas (cada una con su partición de parada temprana)
   se entrenan a la vez como columnas de una misma matriz: el costo por paso es una sola pasada por
   los datos para todas las bolsas.
3. **Explicación exacta.** Como el modelo es aditivo, la contribución de cada variable a la
   predicción de un caso es exactamente :math:`f_j(x_j)` (centrada): no hace falta SHAP por muestreo
   ni un fondo de casos reales (método ``contribuciones``).
4. **Incertidumbre de cada función de forma**: la desviación estándar entre bolsas.

Diseño (decisiones documentadas):

* Efectos principales por defecto (``interactions=0``): con pocos positivos, cada interacción es una
  tabla de dos dimensiones estimada con muy pocos eventos por celda; agrega varianza, la selección de
  pares es inestable entre folds y la ecuación pierde la lectura variable por variable. Las
  interacciones existen como opción para el estudio de ablación.
* Estudio de ablación: cada componente propio del paso se puede retirar por separado sin tocar el
  resto del algoritmo: ``paso='voraz'`` reemplaza la programación dinámica por el árbol voraz del EBM
  estándar y ``newton=False`` reemplaza el paso de Newton por un paso de gradiente. Los otros
  componentes se retiran con parámetros: ``n_bags=1`` (sin bolsas), ``validation_size=0`` (sin parada
  temprana), ``reg_lambda=0`` (sin regularización); ``interactions>0`` agrega pares.
* EBMP v2 (``suavizado > 0``): con pocos eventos, cada tramo de una función de forma es ruidoso.
  El suavizado final hace que los tramos vecinos se presten información (suavizador de Whittaker),
  y el intercepto se reajusta para conservar la calibración. En una simulación con la estructura
  del estudio (2412 casos, 12 factores con cero estructural, prevalencia muy distinta por sexo) su
  ganancia de AUC-PR creció al reducir los eventos (de +0,007 con 242 positivos a +0,014–0,027 con
  60). La intensidad la elige la validación interna; 0 reproduce exactamente EBMP v1.
* Intervalos por cuantiles del entrenamiento (como los del EBM estándar); una variable con pocos
  valores distintos (binarias, one-hot, ordinales) usa directamente sus valores.
* Pesos de muestra (``sample_weight``) en el gradiente y la hessiana: así funciona con
  class_weight='balanced' mediante el envoltorio del estudio, y con SMOTE/ADASYN.
* Identificabilidad: al final cada :math:`f_j` se centra (media ponderada 0 en el entrenamiento) y
  su media pasa al intercepto.

Propiedades (demostradas en el capítulo del modelo original):

* Para cada variable, la partición elegida es la óptima entre todas las de a lo sumo L tramos
  contiguos (principio de optimalidad de Bellman).
* El paso de Newton por tramo minimiza la aproximación cuadrática de la pérdida en ese tramo; con
  tasa ``learning_rate`` pequeña y :math:`\\lambda > 0` la pérdida de entrenamiento decrece en cada
  paso (descenso por bloques sobre una función convexa).
* Complejidad: entrenamiento O(R · p · (n·K + K·L·B²)); predicción O(n · p · log B).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, validate_data

_LOGIT_MAX = 30.0


def _sigmoide(z):
    """Función logística numéricamente estable."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _log_perdida(y, z, w):
    """Pérdida logística ponderada media para logits ``z`` (una columna por bolsa).

    Parameters
    ----------
    y : numpy.ndarray de forma (n,)
        Clase 0/1.
    z : numpy.ndarray de forma (n, K)
        Logits.
    w : numpy.ndarray de forma (n, K)
        Pesos (0 = el caso no cuenta para esa bolsa).

    Returns
    -------
    numpy.ndarray de forma (K,)
    """
    perdida = np.logaddexp(0.0, z) - y[:, None] * z            # -log verosimilitud de Bernoulli
    return (perdida * w).sum(0) / np.maximum(w.sum(0), 1e-12)


def cortes_cuantiles(x, max_bins, w=None):
    """Puntos de corte de una variable: sus valores distintos o, si son muchos, cuantiles ponderados.

    Parameters
    ----------
    x : numpy.ndarray de forma (n,)
    max_bins : int
        Número máximo de intervalos.
    w : numpy.ndarray de forma (n,), opcional
        Pesos de muestra: los casos con peso 0 no intervienen y un peso entero equivale a repetir el
        caso.

    Returns
    -------
    numpy.ndarray
        Cortes crecientes; el intervalo de un valor es ``searchsorted(cortes, valor, 'right')``.
    """
    if w is not None:
        x, w = x[w > 0], w[w > 0]
    u = np.unique(x)
    if len(u) <= max_bins:
        return (u[:-1] + u[1:]) / 2.0                          # un intervalo por valor observado
    if w is None:
        q = np.quantile(x, np.linspace(0, 1, max_bins + 1)[1:-1])
    else:                                                       # cuantiles de la distribución ponderada
        orden = np.argsort(x, kind='stable')
        xs, acum = x[orden], np.cumsum(w[orden])
        objetivos = np.linspace(0, 1, max_bins + 1)[1:-1] * acum[-1]
        q = xs[np.minimum(np.searchsorted(acum, objetivos, side='left'), len(xs) - 1)]
    q = np.unique(q)
    return q[q > u[0]]                                         # sin intervalos vacíos al inicio


def segmentacion_optima(G, H, N, max_leaves, reg_lambda, min_samples_leaf):
    """Paso de Newton óptimo por tramos contiguos (programación dinámica), para K bolsas a la vez.

    Entre todas las particiones de los B intervalos ordenados en a lo sumo ``max_leaves`` tramos
    contiguos con al menos ``min_samples_leaf`` casos cada uno, elige la que maximiza la ganancia
    :math:`\\sum_s G_s^2/(H_s+\\lambda)` y devuelve, para cada intervalo, el valor de Newton de su
    tramo, :math:`-G_s/(H_s+\\lambda)`.

    Parameters
    ----------
    G, H, N : numpy.ndarray de forma (K, B)
        Suma de gradientes, de hessianas y número de casos de entrenamiento por intervalo y bolsa.
    max_leaves : int
        Número máximo de tramos (L).
    reg_lambda : float
        Regularización L2 del valor de cada tramo (en unidades de hessiana).
    min_samples_leaf : int
        Casos mínimos por tramo.

    Returns
    -------
    numpy.ndarray de forma (K, B)
        Valor del paso para cada intervalo (0 si ninguna partición es válida).

    Notes
    -----
    Recurrencia de Bellman: ``V[l, e] = max_{s<e} V[l-1, s] + g(s, e)``, con ``g(s, e)`` la ganancia
    del tramo de intervalos ``[s, e)``. Costo O(K · L · B²).
    """
    K, B = G.shape
    cero = np.zeros((K, 1))
    Gc = np.concatenate([cero, np.cumsum(G, 1)], 1)            # sumas acumuladas (K, B+1)
    Hc = np.concatenate([cero, np.cumsum(H, 1)], 1)
    Nc = np.concatenate([cero, np.cumsum(N, 1)], 1)
    Gs = Gc[:, None, :] - Gc[:, :, None]                        # tramo [s, e): índice (K, s, e)
    Hs = Hc[:, None, :] - Hc[:, :, None]
    Ns = Nc[:, None, :] - Nc[:, :, None]
    s_idx, e_idx = np.arange(B + 1)[:, None], np.arange(B + 1)[None, :]
    valido = (e_idx > s_idx)[None] & (Ns >= min_samples_leaf) & (Hs > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ganancia = np.where(valido, Gs ** 2 / (Hs + reg_lambda), -np.inf)

    V = [ganancia[:, 0, :]]                                     # un tramo: [0, e)
    arg = [None]
    for _ in range(1, max_leaves):
        cand = V[-1][:, :, None] + ganancia                      # (K, s, e)
        arg.append(np.argmax(cand, axis=1))
        V.append(np.max(cand, axis=1))
    finales = np.stack([v[:, B] for v in V], 1)                 # (K, L): ganancia total con l tramos
    paso = np.zeros((K, B))
    for k in range(K):
        l = int(np.argmax(finales[k]))
        if not np.isfinite(finales[k, l]):
            continue                                            # ningún tramo válido: no se actualiza
        e = B
        while True:                                             # reconstrucción de los tramos
            s = 0 if l == 0 else int(arg[l][k, e])
            g, h = Gc[k, e] - Gc[k, s], Hc[k, e] - Hc[k, s]
            paso[k, s:e] = -g / (h + reg_lambda)
            if l == 0:
                break
            e, l = s, l - 1
    return paso


def suavizar_curva(g, W, kappa):
    """Suavizado de una función de forma ordenada: los tramos vecinos se prestan información.

    Resuelve el problema de mínimos cuadrados penalizados (suavizador de Whittaker; P-splines de
    orden 0 de Eilers y Marx, 1996)

    .. math::

        \\min_f \\sum_b W_b (f_b - g_b)^2 + \\gamma \\sum_b (f_b - f_{b-1})^2,
        \\qquad \\gamma = \\kappa \\cdot \\bar W,

    cuya solución es el sistema lineal tridiagonal :math:`(\\operatorname{diag}(W) + \\gamma D^\\top D) f = W g`,
    con :math:`D` la matriz de primeras diferencias. Un tramo con pocos casos (W pequeño) se acerca
    más a sus vecinos; uno con muchos casos casi no cambia. ``kappa = 0`` devuelve ``g`` sin cambios.

    Parameters
    ----------
    g : numpy.ndarray de forma (B,)
        Valores de la función por tramo (aprendidos por el boosting).
    W : numpy.ndarray de forma (B,)
        Peso de cada tramo (suma de pesos de los casos de entrenamiento).
    kappa : float
        Intensidad del suavizado, relativa al peso medio por tramo (≥ 0).

    Returns
    -------
    numpy.ndarray de forma (B,)
    """
    g = np.asarray(g, float)
    if kappa <= 0 or g.size <= 2:
        return g.copy()
    W = np.asarray(W, float) + 1e-9
    D = np.diff(np.eye(g.size), axis=0)
    return np.linalg.solve(np.diag(W) + kappa * W.mean() * (D.T @ D), W * g)


def segmentacion_voraz(G, H, N, max_leaves, reg_lambda, min_samples_leaf):
    """Paso por tramos contiguos elegidos de forma voraz (variante de ablación del EBM estándar).

    Reproduce la lógica de un árbol pequeño construido «primero el mejor» (como el EBM de InterpretML):
    se parte de un solo tramo y, mientras haya menos de ``max_leaves`` tramos, se aplica el corte
    (de cualquier tramo) que más aumenta la ganancia de Newton. Cada decisión es localmente óptima,
    pero el resultado puede no serlo: la ganancia nunca supera a la de ``segmentacion_optima``.

    Parameters
    ----------
    G, H, N : numpy.ndarray de forma (K, B)
        Suma de gradientes, de hessianas y número de casos de entrenamiento por intervalo y bolsa.
    max_leaves : int
        Número máximo de tramos (L).
    reg_lambda : float
        Regularización L2 del valor de cada tramo.
    min_samples_leaf : int
        Casos mínimos por tramo.

    Returns
    -------
    numpy.ndarray de forma (K, B)
        Valor del paso para cada intervalo (0 si ni siquiera un tramo único es válido).

    Notes
    -----
    Costo O(K · L · B) por paso, frente a O(K · L · B²) de la programación dinámica.
    """
    K, B = G.shape
    cero = np.zeros((K, 1))
    Gc = np.concatenate([cero, np.cumsum(G, 1)], 1)
    Hc = np.concatenate([cero, np.cumsum(H, 1)], 1)
    Nc = np.concatenate([cero, np.cumsum(N, 1)], 1)
    paso = np.zeros((K, B))
    for k in range(K):
        def gan(a, b):
            """Ganancia de Newton del tramo [a, b) (−inf si no es válido)."""
            g, h, n = Gc[k, b] - Gc[k, a], Hc[k, b] - Hc[k, a], Nc[k, b] - Nc[k, a]
            return g * g / (h + reg_lambda) if (n >= min_samples_leaf and h > 0) else -np.inf

        if not np.isfinite(gan(0, B)):
            continue                                            # ni un tramo único es válido
        tramos = [(0, B)]
        while len(tramos) < max_leaves:
            mejor = (0.0, None, None)                           # (aumento de ganancia, tramo, corte)
            for i, (a, b) in enumerate(tramos):
                base = gan(a, b)
                for c in range(a + 1, b):
                    aumento = gan(a, c) + gan(c, b) - base
                    if aumento > mejor[0]:
                        mejor = (aumento, i, c)
            if mejor[1] is None:
                break                                           # ningún corte mejora la ganancia
            a, b = tramos.pop(mejor[1])
            tramos += [(a, mejor[2]), (mejor[2], b)]
        for a, b in tramos:
            paso[k, a:b] = -(Gc[k, b] - Gc[k, a]) / (Hc[k, b] - Hc[k, a] + reg_lambda)
    return paso


def _paso_binario(G, H, N, max_leaves, reg_lambda, min_samples_leaf):
    """Paso óptimo para una variable de dos intervalos (caso particular cerrado de la segmentación).

    Compara un solo tramo (desplazamiento común) con dos tramos (un valor por intervalo) y elige el
    de mayor ganancia de Newton; equivale a ``segmentacion_optima`` con B = 2, sin la tabla O(B²).

    Returns
    -------
    numpy.ndarray de forma (K, 2)
    """
    Gt, Ht, Nt = G.sum(1), H.sum(1), N.sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        uno = np.where((Nt >= min_samples_leaf) & (Ht > 0), Gt ** 2 / (Ht + reg_lambda), -np.inf)
        ok2 = (N >= min_samples_leaf).all(1) & (H > 0).all(1) & (max_leaves >= 2)
        dos = np.where(ok2, (G ** 2 / (H + reg_lambda)).sum(1), -np.inf)
        v_uno = -Gt / (Ht + reg_lambda)
        v_dos = -G / (H + reg_lambda)
    paso = np.zeros_like(G)
    usa_dos = ok2 & (dos >= uno)
    usa_uno = ~usa_dos & np.isfinite(uno)
    paso[usa_dos] = v_dos[usa_dos]
    paso[usa_uno] = v_uno[usa_uno, None]
    return paso


class EBMP(ClassifierMixin, BaseEstimator):
    """Máquina de boosting explicable (efectos principales y, opcionalmente, pares), desde cero.

    Parameters
    ----------
    max_bins : int, default=32
        Intervalos máximos por variable (cuantiles del entrenamiento).
    learning_rate : float, default=0.05
        Fracción del paso de Newton que se aplica en cada actualización, en (0, 1].
    max_leaves : int, default=3
        Tramos contiguos máximos de cada actualización de una función de forma.
    min_samples_leaf : int, default=10
        Casos mínimos de entrenamiento por tramo.
    reg_lambda : float, default=1.0
        Regularización L2 del valor de cada tramo.
    max_rounds : int, default=2000
        Rondas máximas (cada ronda recorre todas las variables).
    early_stopping_rounds : int, default=50
        Rondas sin mejora de la pérdida de validación tras las que se detiene una bolsa.
    early_stopping_tolerance : float, default=1e-5
        Mejora mínima de la pérdida de validación (log-pérdida media) que cuenta como mejora.
    validation_size : float, default=0.15
        Fracción estratificada de cada bolsa reservada para la parada temprana (0 = sin parada).
    n_bags : int, default=8
        Bolsas (particiones distintas de validación); las funciones de forma se promedian.
    interactions : int, default=0
        Pares de variables a modelar tras los efectos principales (0 = solo efectos principales).
    interaction_bins : int, default=8
        Intervalos por variable en las tablas de pares.
    random_state : int, default=42
        Semilla de las particiones de validación.
    paso : {'optimo', 'voraz'}, default='optimo'
        Cómo se eligen los tramos de cada actualización: partición óptima por programación
        dinámica (EBMP) o árbol voraz «primero el mejor» (EBM estándar). 'voraz' existe solo para
        el estudio de ablación.
    newton : bool, default=True
        True: paso de Newton (gradiente / hessiana). False: paso de gradiente (gradiente / peso de
        los casos), solo para el estudio de ablación.
    suavizado : float, default=0.0
        EBMP v2. Intensidad κ del suavizado final de cada función de forma ordenada (más de 2
        tramos): los tramos vecinos se prestan información (``suavizar_curva``), lo que reduce la
        varianza cuando cada tramo tiene pocos eventos. Después se reajusta el intercepto para que la
        probabilidad media de entrenamiento siga igual a la proporción observada. 0 = EBMP v1.

    Attributes
    ----------
    intercepto_ : float
    terminos_ : list of str
        Nombre de cada término (variable o par «a & b»).
    funciones_ : list of numpy.ndarray
        Valor de cada término por intervalo (o por celda en los pares), ya centrado.
    funciones_de_ : list of numpy.ndarray
        Desviación estándar entre bolsas de cada función.
    cortes_ : list of numpy.ndarray
        Cortes de cada variable.
    importancias_ : numpy.ndarray
        Media ponderada de |f(x)| de cada término en el entrenamiento.
    rondas_ : numpy.ndarray
        Ronda con la mejor pérdida de validación de cada bolsa (efectos principales).
    historial_ : numpy.ndarray de forma (rondas, n_bags)
        Pérdida de validación por ronda y bolsa (efectos principales).
    """

    def __init__(self, max_bins=32, learning_rate=0.05, max_leaves=3, min_samples_leaf=10,
                 reg_lambda=1.0, max_rounds=2000, early_stopping_rounds=50, early_stopping_tolerance=1e-5,
                 validation_size=0.15, n_bags=8, interactions=0, interaction_bins=8, random_state=42,
                 paso='optimo', newton=True, suavizado=0.0):
        """Guarda los hiperparámetros tal como llegan (convención de scikit-learn: se validan en fit)."""
        self.max_bins = max_bins
        self.learning_rate = learning_rate
        self.max_leaves = max_leaves
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.max_rounds = max_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_tolerance = early_stopping_tolerance
        self.validation_size = validation_size
        self.n_bags = n_bags
        self.interactions = interactions
        self.interaction_bins = interaction_bins
        self.random_state = random_state
        self.paso = paso
        self.newton = newton
        self.suavizado = suavizado

    def __sklearn_tags__(self):
        """Etiquetas de scikit-learn: clasificador binario que no admite vacíos."""
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.input_tags.allow_nan = False
        return tags

    # ------------------------------------------------------------------ validación
    def _validar_hiperparametros(self):
        """Comprueba los hiperparámetros y lanza ValueError con un mensaje claro si alguno es inválido."""
        reglas = [
            (isinstance(self.max_bins, (int, np.integer)) and self.max_bins >= 2, 'max_bins debe ser un entero ≥ 2'),
            (0 < float(self.learning_rate) <= 1, 'learning_rate debe estar en (0, 1]'),
            (isinstance(self.max_leaves, (int, np.integer)) and self.max_leaves >= 1, 'max_leaves debe ser un entero ≥ 1'),
            (int(self.min_samples_leaf) >= 1, 'min_samples_leaf debe ser ≥ 1'),
            (float(self.reg_lambda) >= 0, 'reg_lambda debe ser ≥ 0'),
            (int(self.max_rounds) >= 1, 'max_rounds debe ser ≥ 1'),
            (int(self.early_stopping_rounds) >= 1, 'early_stopping_rounds debe ser ≥ 1'),
            (float(self.early_stopping_tolerance) >= 0, 'early_stopping_tolerance debe ser ≥ 0'),
            (0 <= float(self.validation_size) < 0.5, 'validation_size debe estar en [0, 0,5)'),
            (int(self.n_bags) >= 1, 'n_bags debe ser ≥ 1'),
            (int(self.interactions) >= 0, 'interactions debe ser ≥ 0'),
            (int(self.interaction_bins) >= 2, 'interaction_bins debe ser ≥ 2'),
            (self.paso in ('optimo', 'voraz'), "paso debe ser 'optimo' o 'voraz'"),
            (isinstance(self.newton, (bool, np.bool_)), 'newton debe ser True o False'),
            (float(self.suavizado) >= 0 and np.isfinite(float(self.suavizado)), 'suavizado debe ser ≥ 0'),
        ]
        for ok, mensaje in reglas:
            if not ok:
                raise ValueError(mensaje)

    def _particiones(self, y, rng):
        """Máscaras de entrenamiento y validación de cada bolsa (estratificadas por clase).

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Matrices booleanas (n, K): entrenamiento y validación.
        """
        n, K = len(y), int(self.n_bags)
        val = np.zeros((n, K), bool)
        if self.validation_size > 0:
            for k in range(K):
                for c in (0, 1):
                    idx = np.flatnonzero((y == c) & self._con_peso)
                    m = int(round(self.validation_size * len(idx)))
                    if 0 < m < len(idx):
                        val[rng.choice(idx, m, replace=False), k] = True
        return ~val, val

    # ------------------------------------------------------------------ núcleo
    def _boosting(self, terminos, n_celdas, ordenado, y, w, ent, val, Z0):
        """Boosting cíclico de Newton de un conjunto de términos, con K bolsas a la vez.

        Parameters
        ----------
        terminos : list of numpy.ndarray de forma (n,)
            Índice de intervalo (o celda) de cada caso en cada término.
        n_celdas : list of int
            Número de intervalos (o celdas) de cada término.
        ordenado : list of bool
            True si los intervalos están ordenados (se usa la segmentación óptima); False para las
            tablas de pares (paso de Newton por celda).
        y : numpy.ndarray de forma (n,)
        w : numpy.ndarray de forma (n,)
            Pesos de muestra.
        ent, val : numpy.ndarray de forma (n, K)
            Máscaras de entrenamiento y validación de cada bolsa.
        Z0 : numpy.ndarray de forma (n, K)
            Logits iniciales (intercepto y, en la etapa de pares, los efectos principales).

        Returns
        -------
        (list of numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
            Funciones por término de forma (K, celdas), logits finales (n, K), mejor ronda por
            bolsa e historial de la pérdida de validación.
        """
        K, n = Z0.shape[1], Z0.shape[0]
        Wt_all, Wv = w[:, None] * ent, w[:, None] * val
        Nt_all = ent * (w[:, None] > 0)                          # casos (con peso) por bolsa
        hay_val = bool(val.any())
        F = [np.zeros((K, b)) for b in n_celdas]
        Z = Z0.copy()
        mejor = np.full(K, np.inf)
        mejor_F = [f.copy() for f in F]
        mejor_ronda = np.zeros(K, int)
        sin_mejora = np.zeros(K, int)
        activa = np.ones(K, bool)
        historial = []
        lr, lam, msl, L = float(self.learning_rate), float(self.reg_lambda), int(self.min_samples_leaf), int(self.max_leaves)
        tol = float(self.early_stopping_tolerance)
        newton = bool(self.newton)
        segmentar = segmentacion_optima if self.paso == 'optimo' else segmentacion_voraz

        def preparar(act):
            """Arreglos de trabajo solo con las bolsas activas (las detenidas ya no se recalculan)."""
            k = len(act)
            fil = np.arange(k)
            idx = [(bins[:, None] + B * fil[None, :]).ravel() for bins, B in zip(terminos, n_celdas)]
            cnt = [np.bincount(ix, weights=Nt_all[:, act].ravel(), minlength=k * B).reshape(k, B)
                   for ix, B in zip(idx, n_celdas)]
            return k, idx, cnt, Wt_all[:, act]

        act = np.flatnonzero(activa)
        k, indices, conteos, Wt = preparar(act)
        Zw = Z[:, act].copy()
        for ronda in range(1, int(self.max_rounds) + 1):
            for t, B in enumerate(n_celdas):
                p = _sigmoide(Zw)
                g = ((p - y[:, None]) * Wt).ravel()
                h = (p * (1 - p) * Wt).ravel() if newton else Wt.ravel()   # Newton o gradiente
                G = np.bincount(indices[t], weights=g, minlength=k * B).reshape(k, B)
                H = np.bincount(indices[t], weights=h, minlength=k * B).reshape(k, B)
                N = conteos[t]
                if not ordenado[t]:                              # tabla de un par: Newton por celda
                    paso = np.where(N >= msl, -G / (H + lam + 1e-12), 0.0)
                elif B == 2:                                     # variable binaria: forma cerrada
                    paso = _paso_binario(G, H, N, L, lam, msl)
                else:                                            # segmentación óptima (Bellman)
                    paso = segmentar(G, H, N, L, lam, msl)
                F[t][act] += lr * paso
                Zw += lr * paso.ravel()[indices[t]].reshape(n, k)
                np.clip(Zw, -_LOGIT_MAX, _LOGIT_MAX, out=Zw)
            Z[:, act] = Zw
            if not hay_val:
                mejor_F, mejor_ronda[:] = [f.copy() for f in F], ronda
                continue
            perdida = np.full(K, np.inf)
            perdida[act] = _log_perdida(y, Zw, Wv[:, act])
            historial.append(np.where(activa, perdida, np.nan))
            mejora = activa & (perdida < mejor - tol)            # mejora mínima (evita rondas inútiles)
            if mejora.any():
                mejor[mejora] = perdida[mejora]
                mejor_ronda[mejora] = ronda
                for t in range(len(F)):
                    mejor_F[t][mejora] = F[t][mejora]
            sin_mejora = np.where(mejora, 0, sin_mejora + 1)
            nueva = activa & (sin_mejora < int(self.early_stopping_rounds))
            if not nueva.any():
                break
            if (nueva != activa).any():                          # se detuvo alguna bolsa: menos columnas
                activa = nueva
                act = np.flatnonzero(activa)
                k, indices, conteos, Wt = preparar(act)
                Zw = Z[:, act].copy()
        Zf = Z0.copy()
        filas = np.arange(K)
        for t, bins in enumerate(terminos):
            Zf += mejor_F[t][filas[None, :], bins[:, None]]
        return mejor_F, Zf, mejor_ronda, np.array(historial)

    def _pares(self, Xb, y, w, Z):
        """Elige los ``interactions`` pares con mayor ganancia de Newton sobre el residuo.

        Returns
        -------
        list of (int, int)
        """
        p = _sigmoide(Z.mean(1))
        g, h = (p - y) * w, p * (1 - p) * w
        cand = []
        nb = [int(self.n_cortes_par_[j].size + 1) for j in range(Xb.shape[1])]
        for a in range(Xb.shape[1]):
            for b in range(a + 1, Xb.shape[1]):
                if nb[a] < 2 or nb[b] < 2:
                    continue
                celda = self._celda_par(a, b)
                G = np.bincount(celda, weights=g, minlength=nb[a] * nb[b])
                H = np.bincount(celda, weights=h, minlength=nb[a] * nb[b])
                cand.append(((G ** 2 / (H + self.reg_lambda + 1e-12)).sum(), a, b))
        cand.sort(reverse=True)
        return [(a, b) for _, a, b in cand[:int(self.interactions)]]

    def _celda_par(self, a, b, Xb=None):
        """Índice de celda de la tabla del par (a, b) en la malla gruesa de cada caso."""
        Xb = self._Xb_par if Xb is None else Xb
        nb_b = int(self.n_cortes_par_[b].size + 1)
        return Xb[:, a] * nb_b + Xb[:, b]

    # ------------------------------------------------------------------ API de scikit-learn
    def fit(self, X, y, sample_weight=None):
        """Ajusta el modelo.

        Parameters
        ----------
        X : array-like de forma (n, p)
            Predictoras numéricas (en el estudio, la salida del ColumnTransformer).
        y : array-like de forma (n,)
            Clase binaria.
        sample_weight : array-like de forma (n,), opcional
            Pesos de muestra (class_weight='balanced' en el estudio).

        Returns
        -------
        EBMP
            El propio objeto ajustado.

        Raises
        ------
        ValueError
            Si hay vacíos o infinitos, si ``y`` no tiene exactamente dos clases o si algún
            hiperparámetro es inválido.
        """
        self._validar_hiperparametros()
        X, y = validate_data(self, X, y, dtype=np.float64, ensure_min_samples=2)
        check_classification_targets(y)                          # rechaza objetivos continuos
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError('Only binary classification is supported. EBMP necesita exactamente 2 clases '
                             f'en y (recibió {len(self.classes_)}).')
        y01 = (y == self.classes_[1]).astype(float)
        w = _check_sample_weight(sample_weight, X, dtype=np.float64)
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError('sample_weight debe ser no negativo y no todo cero')
        w = w * (len(w) / w.sum())                                # escala media 1 (no cambia el óptimo)
        rng = np.random.default_rng(self.random_state)
        n, p = X.shape

        # intervalos de cada variable
        self.cortes_ = [cortes_cuantiles(X[:, j], int(self.max_bins), w) for j in range(p)]
        Xb = np.column_stack([np.searchsorted(c, X[:, j], side='right') for j, c in enumerate(self.cortes_)])
        n_bins = [c.size + 1 for c in self.cortes_]

        # bolsas e intercepto inicial (log-odds ponderado de cada bolsa)
        self._con_peso = w > 0
        ent, val = self._particiones(y01, rng)
        del self._con_peso
        Wt = w[:, None] * ent
        tasa = np.clip((Wt * y01[:, None]).sum(0) / Wt.sum(0), 1e-6, 1 - 1e-6)
        b0 = np.log(tasa / (1 - tasa))
        Z0 = np.repeat(b0[None, :], n, 0)

        # etapa 1: efectos principales
        F, Z, self.rondas_, self.historial_ = self._boosting(
            [Xb[:, j] for j in range(p)], n_bins, [True] * p, y01, w, ent, val, Z0)
        terminos, funciones, desv = list(self._nombres(p)), [], []
        for f in F:
            funciones.append(f.mean(0))
            desv.append(f.std(0, ddof=1) if f.shape[0] > 1 else np.zeros(f.shape[1]))
        self.pares_ = []

        # etapa 2 (opcional): pares sobre el residuo de los efectos principales
        if int(self.interactions) > 0 and p >= 2:
            self.n_cortes_par_ = [cortes_cuantiles(X[:, j], int(self.interaction_bins), w) for j in range(p)]
            self._Xb_par = np.column_stack([np.searchsorted(c, X[:, j], side='right')
                                            for j, c in enumerate(self.n_cortes_par_)])
            self.pares_ = self._pares(Xb, y01, w, Z)
            if self.pares_:
                celdas = [self._celda_par(a, b) for a, b in self.pares_]
                n_c = [(self.n_cortes_par_[a].size + 1) * (self.n_cortes_par_[b].size + 1) for a, b in self.pares_]
                Fp, Z, _, _ = self._boosting(celdas, n_c, [False] * len(celdas), y01, w, ent, val, Z)
                nombres = self._nombres(p)
                for (a, b), f in zip(self.pares_, Fp):
                    terminos.append(f'{nombres[a]} & {nombres[b]}')
                    funciones.append(f.mean(0))
                    desv.append(f.std(0, ddof=1) if f.shape[0] > 1 else np.zeros(f.shape[1]))
            del self._Xb_par

        # EBMP v2: suavizado final de las funciones ordenadas (los pares no se suavizan)
        kappa = float(self.suavizado)
        if kappa > 0:
            for j in range(p):
                if n_bins[j] > 2:
                    W = np.bincount(Xb[:, j], weights=w, minlength=n_bins[j])
                    funciones[j] = suavizar_curva(funciones[j], W, kappa)

        # identificabilidad: cada término con media ponderada 0; su media pasa al intercepto
        self.intercepto_ = float(b0.mean())
        indices = [Xb[:, j] for j in range(p)] + ([self._celda_par(a, b, self._binear_par(X)) for a, b in self.pares_]
                                                 if self.pares_ else [])
        imp = []
        for t, idx in enumerate(indices):
            media = float(np.average(funciones[t][idx], weights=w))
            funciones[t] = funciones[t] - media
            self.intercepto_ += media
            imp.append(float(np.average(np.abs(funciones[t][idx]), weights=w)))
        if kappa > 0:                       # el suavizado cambia la media de p: se reajusta el intercepto
            z = self.intercepto_ + sum(funciones[t][idx] for t, idx in enumerate(indices))
            c = 0.0
            for _ in range(25):             # Newton en una dimensión: Σ w (σ(z + c) − y) = 0
                q = _sigmoide(np.clip(z + c, -_LOGIT_MAX, _LOGIT_MAX))
                g1, h1 = float(np.sum(w * (q - y01))), float(np.sum(w * q * (1 - q)))
                if h1 <= 0 or abs(g1) < 1e-10:
                    break
                c -= g1 / h1
            self.intercepto_ += c
        self.terminos_, self.funciones_, self.funciones_de_ = terminos, funciones, desv
        self.importancias_ = np.array(imp)
        return self

    def _binear_par(self, X):
        """Índices de la malla gruesa de los pares para una matriz X."""
        return np.column_stack([np.searchsorted(c, X[:, j], side='right') for j, c in enumerate(self.n_cortes_par_)])

    def _nombres(self, p):
        """Nombres de las variables (los de entrada si se conocen)."""
        nombres = getattr(self, 'feature_names_in_', None)
        return [str(v) for v in nombres] if nombres is not None else [f'x{j}' for j in range(p)]

    def contribuciones(self, X):
        """Contribución exacta de cada término al logit de cada caso.

        Como el modelo es aditivo, ``decision_function(X) = intercepto_ + contribuciones(X).sum(1)``:
        es una explicación local exacta (equivale a SHAP para un modelo aditivo con términos
        centrados) y no necesita un fondo de casos.

        Parameters
        ----------
        X : array-like de forma (n, p)

        Returns
        -------
        numpy.ndarray de forma (n, n_términos)
        """
        check_is_fitted(self, 'funciones_')
        X = validate_data(self, X, dtype=np.float64, reset=False)
        p = X.shape[1]
        cols = [self.funciones_[j][np.searchsorted(self.cortes_[j], X[:, j], side='right')] for j in range(p)]
        if self.pares_:
            Xp = self._binear_par(X)
            for t, (a, b) in enumerate(self.pares_):
                cols.append(self.funciones_[p + t][self._celda_par(a, b, Xp)])
        return np.column_stack(cols)

    def decision_function(self, X):
        """Logit predicho.

        Parameters
        ----------
        X : array-like de forma (n, p)

        Returns
        -------
        numpy.ndarray de forma (n,)
        """
        check_is_fitted(self, 'funciones_')
        return self.intercepto_ + self.contribuciones(X).sum(1)

    def predict_proba(self, X):
        """Probabilidades de cada clase.

        Parameters
        ----------
        X : array-like de forma (n, p)

        Returns
        -------
        numpy.ndarray de forma (n, 2)
            Columna 1: probabilidad de ``classes_[1]``.
        """
        check_is_fitted(self, 'funciones_')
        p1 = _sigmoide(np.clip(self.decision_function(X), -_LOGIT_MAX, _LOGIT_MAX))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        """Clase predicha con umbral 0,5 (el estudio elige el umbral en X_thresh).

        Parameters
        ----------
        X : array-like de forma (n, p)

        Returns
        -------
        numpy.ndarray de forma (n,)
        """
        check_is_fitted(self, 'funciones_')
        return self.classes_[(self.decision_function(X) > 0).astype(int)]   # logit > 0 ⇔ p > 0,5

    def funciones_forma(self):
        """Tabla legible de las funciones de forma de los efectos principales.

        Returns
        -------
        list of dict
            Por variable: nombre, intervalos (límites inferior y superior), valor de f, DE entre
            bolsas e importancia.
        """
        check_is_fitted(self, 'funciones_')
        filas = []
        for j, c in enumerate(self.cortes_):
            lim = np.concatenate([[-np.inf], c, [np.inf]])
            filas.append({'termino': self.terminos_[j], 'desde': lim[:-1].tolist(), 'hasta': lim[1:].tolist(),
                          'f': self.funciones_[j].tolist(), 'de': self.funciones_de_[j].tolist(),
                          'importancia': float(self.importancias_[j])})
        return filas
