/*
 * Copyright 2020 Eric Medvet <eric.medvet@gmail.com> (as eric)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.units.erallab;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.PhaseSin;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.sensors.AreaRatio;
import it.units.erallab.hmsrobots.core.sensors.TimeFunction;
import it.units.erallab.hmsrobots.core.sensors.Touch;
import it.units.erallab.hmsrobots.core.sensors.Velocity;
import it.units.erallab.hmsrobots.tasks.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.*;
import it.units.malelab.jgea.core.evolver.stopcondition.Births;
import it.units.malelab.jgea.core.listener.Listener;
import it.units.malelab.jgea.core.listener.MultiFileListenerFactory;
import it.units.malelab.jgea.core.listener.collector.*;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.selector.Tournament;
import it.units.malelab.jgea.core.selector.Worst;
import it.units.malelab.jgea.core.util.Misc;
import it.units.malelab.jgea.core.util.Pair;
import it.units.malelab.jgea.distance.Jaccard;
import it.units.malelab.jgea.representation.graph.*;
import it.units.malelab.jgea.representation.graph.numeric.Output;
import it.units.malelab.jgea.representation.graph.numeric.functiongraph.BaseFunction;
import it.units.malelab.jgea.representation.graph.numeric.functiongraph.FunctionGraph;
import it.units.malelab.jgea.representation.graph.numeric.functiongraph.FunctionNode;
import it.units.malelab.jgea.representation.graph.numeric.functiongraph.ShallowSparseFactory;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;
import it.units.malelab.jgea.representation.sequence.numeric.GeometricCrossover;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;
import org.apache.commons.lang3.SerializationUtils;
import org.dyn4j.dynamics.Settings;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author eric
 * @created 2020/08/18
 * @project VSREvolution
 */
public class ControllerComparison extends Worker {

  public ControllerComparison(String[] args) {
    super(args);
  }

  public static void main(String[] args) {
    new ControllerComparison(args);
  }

  private interface IODimMapper extends Function<Grid<? extends SensingVoxel>, Pair<Integer, Integer>> {
  }

  private interface RobotMapper extends Function<Grid<? extends SensingVoxel>, Function<Function<double[], double[]>, Robot<?>>> {
  }

  private interface EvolverMapper extends BiFunction<Pair<IODimMapper, RobotMapper>, Grid<? extends SensingVoxel>, Evolver<?, Robot<?>, Double>> {
  }

  @Override
  public void run() {
    double episodeTime = d(a("episodeT", "10.0"));
    int nBirths = i(a("nBirths", "1000"));
    int[] seeds = ri(a("seed", "0:1"));
    List<String> terrains = l(a("terrain", "flat"));
    List<String> evolverMapperNames = l(a("evolver", "mlp-0.65-cmaes"));
    List<String> bodyNames = l(a("body", "biped-4x3-f"));
    List<String> robotMapperNames = l(a("mapper", "centralized"));
    Locomotion.Metric metric = Locomotion.Metric.TRAVELED_X_DISTANCE;
    //prepare file listeners
    MultiFileListenerFactory<Object, Robot<?>, Double> statsListenerFactory = new MultiFileListenerFactory<>(
        a("dir", "."),
        a("statsFile", null)
    );
    MultiFileListenerFactory<Object, Robot<?>, Double> serializedListenerFactory = new MultiFileListenerFactory<>(
        a("dir", "."),
        a("serializedFile", null)
    );
    Random bodyRandom = new Random(1);
    Map<String, Grid<? extends SensingVoxel>> bodies = bodyNames.stream()
        .collect(Collectors.toMap(n -> n, n -> buildBodyFromName(n, bodyRandom)));
    Map<String, Pair<IODimMapper, RobotMapper>> robotMappers = robotMapperNames.stream()
        .collect(Collectors.toMap(n -> n, ControllerComparison::buildRobotMapperFromName));
    Map<String, EvolverMapper> evolverMappers = evolverMapperNames.stream()
        .collect(Collectors.toMap(n -> n, ControllerComparison::buildEvolverBuilderFromName));
    L.info("Evolvers: " + evolverMappers.keySet());
    L.info("Mappers: " + robotMappers.keySet());
    L.info("Bodies: " + bodies.keySet());
    for (int seed : seeds) {
      for (String terrain : terrains) {
        for (Map.Entry<String, Grid<? extends SensingVoxel>> bodyEntry : bodies.entrySet()) {
          for (Map.Entry<String, Pair<IODimMapper, RobotMapper>> robotMapperEntry : robotMappers.entrySet()) {
            for (Map.Entry<String, EvolverMapper> evolverMapperEntry : evolverMappers.entrySet()) {
              Map<String, String> keys = new TreeMap<>(Map.of(
                  "seed", Integer.toString(seed),
                  "terrain", terrain,
                  "body", bodyEntry.getKey(),
                  "mapper", robotMapperEntry.getKey(),
                  "evolver", evolverMapperEntry.getKey()
              ));
              Grid<? extends SensingVoxel> body = bodyEntry.getValue();
              Evolver<?, Robot<?>, Double> evolver = evolverMapperEntry.getValue().apply(robotMapperEntry.getValue(), body);
              Locomotion locomotion = new Locomotion(
                  episodeTime,
                  Locomotion.createTerrain(terrain),
                  Lists.newArrayList(metric),
                  new Settings()
              );
              List<DataCollector<?, ? super Robot<SensingVoxel>, ? super Double>> collectors = List.of(
                  new Static(keys),
                  new Basic(),
                  new Population(),
                  new Diversity(),
                  new BestInfo("%7.5f")
              );
              Listener<? super Object, ? super Robot<?>, ? super Double> listener;
              if (statsListenerFactory.getBaseFileName() == null) {
                listener = listener(collectors.toArray(DataCollector[]::new));
              } else {
                listener = statsListenerFactory.build(collectors.toArray(DataCollector[]::new));
              }
              if (serializedListenerFactory.getBaseFileName() != null) {
                listener = serializedListenerFactory.build(
                    new Static(keys),
                    new Basic(),
                    new FunctionOfOneBest<>(i -> List.of(
                        new Item("fitness.value", i.getFitness(), "%7.5f"),
                        new Item("serialized.robot", Utils.safelySerialize(i.getSolution()), "%s"),
                        new Item("serialized.genotype", Utils.safelySerialize((Serializable) i.getGenotype()), "%s")
                    ))
                ).then(listener);
              }
              try {
                Stopwatch stopwatch = Stopwatch.createStarted();
                L.info(String.format("Starting %s", keys));
                Collection<Robot<?>> solutions = evolver.solve(
                    Misc.cached(robot -> locomotion.apply(robot).get(0), 10000),
                    new Births(nBirths),
                    new Random(seed),
                    executorService,
                    Listener.onExecutor(
                        listener,
                        executorService
                    )
                );
                L.info(String.format("Done %s: %d solutions in %4ds",
                    keys,
                    solutions.size(),
                    stopwatch.elapsed(TimeUnit.SECONDS)
                ));
              } catch (InterruptedException | ExecutionException e) {
                L.severe(String.format("Cannot complete %s due to %s",
                    keys,
                    e
                ));
                e.printStackTrace();
              }
            }
          }
        }
      }
    }
  }

  private static Pair<IODimMapper, RobotMapper> buildRobotMapperFromName(String name) {
    if (name.matches("centralized")) {
      return Pair.of(
          body -> Pair.of(CentralizedSensing.nOfInputs(body), CentralizedSensing.nOfOutputs(body)),
          body -> f -> new Robot<>(
              new CentralizedSensing(body, f),
              SerializationUtils.clone(body)
          )
      );
    }
    if (name.matches("phases-(?<f>\\d+(\\.\\d+)?)")) {
      return Pair.of(
          body -> Pair.of(2, 1),
          body -> f -> new Robot<>(
              new PhaseSin(
                  -Double.parseDouble(paramValue("phases-(?<f>\\d+(\\.\\d+)?)", name, "f")),
                  1d,
                  Grid.create(
                      body.getW(),
                      body.getH(),
                      (x, y) -> f.apply(new double[]{(double) x / (double) body.getW(), (double) y / (double) body.getH()})[0]
                  )),
              SerializationUtils.clone(body)
          )
      );
    }
    throw new IllegalArgumentException(String.format("Unknown mapper name: %s", name));
  }

  private static Grid<? extends SensingVoxel> buildBodyFromName(String name, Random random) {
    if (name.matches("biped-(?<w>\\d+)x(?<h>\\d+)-(?<cgp>[tf])")) {
      int w = Integer.parseInt(paramValue("biped-(?<w>\\d+)x(?<h>\\d+)-(?<cgp>[tf])", name, "w"));
      int h = Integer.parseInt(paramValue("biped-(?<w>\\d+)x(?<h>\\d+)-(?<cgp>[tf])", name, "h"));
      boolean withCentralPatternGenerator = paramValue("biped-(?<w>\\d+)x(?<h>\\d+)-(?<cgp>[tf])", name, "cgp").equals("t");
      return Grid.create(
          w, h,
          (x, y) -> (y == 0 && x > 0 && x < w - 1) ? null : new SensingVoxel(ofNonNull(
              new AreaRatio(),
              (y == 0) ? new Touch() : null,
              (y == h - 1) ? new Velocity(true, 3d, Velocity.Axis.X, Velocity.Axis.Y) : null,
              (x == w - 1 && y == h - 1 && withCentralPatternGenerator) ? new TimeFunction(t -> Math.sin(2 * Math.PI * -1 * t), -1, 1) : null
          ).stream().filter(Objects::nonNull).collect(Collectors.toList())));
    }
    throw new IllegalArgumentException(String.format("Unknown body name: %s", name));
  }

  private static EvolverMapper buildEvolverBuilderFromName(String name) {
    PartialComparator<Individual<?, Robot<?>, Double>> comparator = PartialComparator.from(Double.class).reversed().comparing(Individual::getFitness);
    if (name.matches("mlp-(?<h>\\d+(\\.\\d+)?)-ga-(?<nPop>\\d+)")) {
      double ratioOfFirstLayer = Double.parseDouble(paramValue("mlp-(?<h>\\d+(\\.\\d+)?)-ga-(?<nPop>\\d+)", name, "h"));
      int nPop = Integer.parseInt(paramValue("mlp-(?<h>\\d+(\\.\\d+)?)-ga-(?<nPop>\\d+)", name, "nPop"));
      return (p, body) -> new StandardEvolver<>(
          ((Function<List<Double>, MultiLayerPerceptron>) ws -> new MultiLayerPerceptron(
              MultiLayerPerceptron.ActivationFunction.TANH,
              p.first().apply(body).first(),
              ratioOfFirstLayer == 0 ? new int[0] : new int[]{(int) Math.round(p.first().apply(body).first() * ratioOfFirstLayer)},
              p.first().apply(body).second(),
              ws.stream().mapToDouble(d -> d).toArray()
          )).andThen(mlp -> p.second().apply(body).apply(mlp)),
          new FixedLengthListFactory<>(
              MultiLayerPerceptron.countWeights(
                  p.first().apply(body).first(),
                  ratioOfFirstLayer == 0 ? new int[0] : new int[]{(int) Math.round(p.first().apply(body).first() * ratioOfFirstLayer)},
                  p.first().apply(body).second()
              ),
              new UniformDoubleFactory(-1, 1)
          ),
          comparator,
          nPop,
          Map.of(
              new GaussianMutation(1d), 0.2d,
              new GeometricCrossover(), 0.8d
          ),
          new Tournament(5),
          new Worst(),
          nPop,
          true
      );
    }
    if (name.matches("mlp-(?<h>\\d+(\\.\\d+)?)-gadiv-(?<nPop>\\d+)")) {
      double ratioOfFirstLayer = Double.parseDouble(paramValue("mlp-(?<h>\\d+(\\.\\d+)?)-gadiv-(?<nPop>\\d+)", name, "h"));
      int nPop = Integer.parseInt(paramValue("mlp-(?<h>\\d+(\\.\\d+)?)-gadiv-(?<nPop>\\d+)", name, "nPop"));
      return (p, body) -> new StandardWithEnforcedDiversityEvolver<>(
          ((Function<List<Double>, MultiLayerPerceptron>) ws -> new MultiLayerPerceptron(
              MultiLayerPerceptron.ActivationFunction.TANH,
              p.first().apply(body).first(),
              ratioOfFirstLayer == 0 ? new int[0] : new int[]{(int) Math.round(p.first().apply(body).first() * ratioOfFirstLayer)},
              p.first().apply(body).second(),
              ws.stream().mapToDouble(d -> d).toArray()
          )).andThen(mlp -> p.second().apply(body).apply(mlp)),
          new FixedLengthListFactory<>(
              MultiLayerPerceptron.countWeights(
                  p.first().apply(body).first(),
                  ratioOfFirstLayer == 0 ? new int[0] : new int[]{(int) Math.round(p.first().apply(body).first() * ratioOfFirstLayer)},
                  p.first().apply(body).second()
              ),
              new UniformDoubleFactory(-1, 1)
          ),
          comparator,
          nPop,
          Map.of(
              new GaussianMutation(1d), 0.2d,
              new GeometricCrossover(), 0.8d
          ),
          new Tournament(5),
          new Worst(),
          nPop,
          true,
          100
      );
    }
    if (name.matches("mlp-(?<h>\\d+(\\.\\d+)?)-cmaes")) {
      double ratioOfFirstLayer = Double.parseDouble(paramValue("mlp-(?<h>\\d+(\\.\\d+)?)-cmaes", name, "h"));
      return (p, body) -> new CMAESEvolver<>(
          ((Function<List<Double>, MultiLayerPerceptron>) ws -> new MultiLayerPerceptron(
              MultiLayerPerceptron.ActivationFunction.TANH,
              p.first().apply(body).first(),
              ratioOfFirstLayer == 0 ? new int[0] : new int[]{(int) Math.round(p.first().apply(body).first() * ratioOfFirstLayer)},
              p.first().apply(body).second(),
              ws.stream().mapToDouble(d -> d).toArray()
          )).andThen(mlp -> p.second().apply(body).apply(mlp)),
          new FixedLengthListFactory<>(
              MultiLayerPerceptron.countWeights(
                  p.first().apply(body).first(),
                  ratioOfFirstLayer == 0 ? new int[0] : new int[]{(int) Math.round(p.first().apply(body).first() * ratioOfFirstLayer)},
                  p.first().apply(body).second()
              ),
              new UniformDoubleFactory(-1, 1)
          ),
          comparator,
          -1,
          1
      );
    }
    if (name.matches("fgraph-hash-speciated-(?<nPop>\\d+)")) {
      int nPop = Integer.parseInt(paramValue("fgraph-hash-speciated-(?<nPop>\\d+)", name, "nPop"));
      return (p, body) -> new SpeciatedEvolver<>(
          GraphUtils.mapper((Function<IndexedNode<Node>, Node>) IndexedNode::content, (Function<Collection<Double>, Double>) Misc::first)
              .andThen(FunctionGraph.builder())
              .andThen(fg -> p.second().apply(body).apply(fg)),
          new ShallowSparseFactory(
              0d, 0d, 1d,
              p.first().apply(body).first(),
              p.first().apply(body).second()
          ).then(GraphUtils.mapper(IndexedNode.incrementerMapper(Node.class), Misc::first)),
          comparator,
          nPop,
          Map.of(
              new IndexedNodeAddition<>(
                  FunctionNode.sequentialIndexFactory(BaseFunction.TANH),
                  n -> n.getFunction().hashCode(),
                  p.first().apply(body).first() + p.first().apply(body).second() + 1,
                  (w, r) -> w,
                  (w, r) -> r.nextGaussian()
              ), 1d,
              new ArcModification<>((w, r) -> w + r.nextGaussian(), 1d), 1d,
              new ArcAddition
                  <>(Random::nextGaussian, false), 3d,
              new AlignedCrossover<>(
                  (w1, w2, r) -> w1 + (w2 - w1) * (r.nextDouble() * 3d - 1d),
                  node -> node.content() instanceof Output,
                  false
              ), 1d
          ),
          5,
          (new Jaccard()).on(i -> i.getGenotype().nodes()),
          0.25,
          individuals -> {
            double[] fitnesses = individuals.stream().mapToDouble(Individual::getFitness).toArray();
            Individual<Graph<IndexedNode<Node>, Double>, Robot<?>, Double> r = Misc.first(individuals);
            return new Individual<>(
                r.getGenotype(),
                r.getSolution(),
                Misc.median(fitnesses),
                r.getBirthIteration()
            );
          },
          0.75
      );
    }
    throw new IllegalArgumentException(String.format("Unknown evolver name: %s", name));
  }

  private static String paramValue(String pattern, String string, String paramName) {
    Matcher matcher = Pattern.compile(pattern).matcher(string);
    if (matcher.matches()) {
      return matcher.group(paramName);
    }
    throw new IllegalStateException(String.format("Param %s not found in %s with pattern %s", paramName, string, pattern));
  }

  private static <E> List<E> ofNonNull(E... es) {
    List<E> list = new ArrayList<>();
    for (E e : es) {
      if (e != null) {
        list.add(e);
      }
    }
    return list;
  }
}