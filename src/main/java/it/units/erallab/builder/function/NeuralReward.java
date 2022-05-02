package it.units.erallab.builder.function;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.controllers.rl.ClusteredObservationFunction;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;

/**
 * @author eric
 */
public class NeuralReward implements NamedProvider<PrototypedFunctionBuilder<List<Double>, ToDoubleFunction<Grid<Voxel>>>> {

  protected final MultiLayerPerceptron.ActivationFunction activationFunction;
  protected final Function<Grid<Voxel>, double[]> sensorsExtractor;
  protected final int inputDimension;

  public NeuralReward(MultiLayerPerceptron.ActivationFunction activationFunction, Function<Grid<Voxel>, double[]> sensorsExtractor, int inputDimension) {
    this.activationFunction = activationFunction;
    this.sensorsExtractor = sensorsExtractor;
    this.inputDimension = inputDimension;
  }

  protected static int[] innerNeurons(int nOfInputs, int nOfOutputs, double innerLayerRatio, int nOfInnerLayers) {
    int[] innerNeurons = new int[nOfInnerLayers];
    int centerSize = (int) Math.max(2, Math.round(nOfInputs * innerLayerRatio));
    if (nOfInnerLayers > 1) {
      for (int i = 0; i < nOfInnerLayers / 2; i++) {
        innerNeurons[i] = nOfInputs + (centerSize - nOfInputs) / (nOfInnerLayers / 2 + 1) * (i + 1);
      }
      for (int i = nOfInnerLayers / 2; i < nOfInnerLayers; i++) {
        innerNeurons[i] = centerSize + (nOfOutputs - centerSize) / (nOfInnerLayers / 2 + 1) * (i - nOfInnerLayers / 2);
      }
    } else if (nOfInnerLayers > 0) {
      innerNeurons[0] = centerSize;
    }
    return innerNeurons;
  }

  @Override
  public PrototypedFunctionBuilder<List<Double>, ToDoubleFunction<Grid<Voxel>>> build(Map<String, String> params) {
    double innerLayerRatio = Double.parseDouble(params.getOrDefault("r", "0.65"));
    int nOfInnerLayers = Integer.parseInt(params.getOrDefault("nIL", "1"));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<List<Double>, ToDoubleFunction<Grid<Voxel>>> buildFor(ToDoubleFunction<Grid<Voxel>> function) {
        return values -> {
          int nOfInputs = inputDimension;
          int[] innerNeurons = innerNeurons(nOfInputs, 1, innerLayerRatio, nOfInnerLayers);
          int nOfWeights = MultiLayerPerceptron.countWeights(nOfInputs, innerNeurons, 1);
          if (nOfWeights != values.size()) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of values for weights: %d expected, %d found",
                nOfWeights,
                values.size()
            ));
          }
          return s -> {
            MultiLayerPerceptron mlp = new MultiLayerPerceptron(
                    activationFunction,
                    nOfInputs,
                    innerNeurons,
                    1,
                    values.stream().mapToDouble(d -> d).toArray()
            );
            return mlp.apply(sensorsExtractor.apply(s))[0];
          };
        };
      }

      @Override
      public List<Double> exampleFor(ToDoubleFunction<Grid<Voxel>> function) {
        return Collections.nCopies(
            MultiLayerPerceptron.countWeights(
                MultiLayerPerceptron.countNeurons(
                    inputDimension,
                    innerNeurons(
                        inputDimension,
                        1,
                        innerLayerRatio,
                        nOfInnerLayers
                    ),
                    1
                )
            ),
            0d
        );
      }
    };
  }


}
