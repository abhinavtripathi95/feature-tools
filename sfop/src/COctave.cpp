#include "COctave.h"

namespace SFOP {

void COctave::detect(
        std::list<CFeature>* f_features_p)
{
    const CLayer** l_layers_p = new const CLayer*[m_numLayers + 2];
    for (int l = -1; l < (int) m_numLayers + 1; ++l) {
        std::cout << "Layer " << l << " of octave " << m_octave << std::endl;
        l_layers_p[l + 1] = new CLayer(m_factory_p, m_image_p, l, m_octave, m_numLayers, m_type);
        if (l < 1) continue;
        l_layers_p[l]->detect(f_features_p, l_layers_p[l - 1], l_layers_p[l + 1]);
        delete l_layers_p[l - 1];
        l_layers_p[l - 1] = NULL;
    }
    for (int l = m_numLayers; l < (int) m_numLayers + 1; ++l) {
        delete l_layers_p[l + 1];
        l_layers_p[l + 1] = NULL;
    }
    delete[] l_layers_p;
}

}

