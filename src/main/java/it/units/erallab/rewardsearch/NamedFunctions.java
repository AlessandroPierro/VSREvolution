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

package it.units.erallab.rewardsearch;

import it.units.erallab.hmsrobots.behavior.BehaviorUtils;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.snapshots.VoxelPoly;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.tasks.rllocomotion.RLEnsembleOutcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.viewers.GridFileWriter;
import it.units.erallab.hmsrobots.viewers.NamedValue;
import it.units.erallab.hmsrobots.viewers.VideoUtils;
import it.units.erallab.locomotion.Starter;
import it.units.erallab.locomotion.Starter.ValidationOutcome;
import it.units.malelab.jgea.core.listener.Accumulator;
import it.units.malelab.jgea.core.listener.AccumulatorFactory;
import it.units.malelab.jgea.core.listener.NamedFunction;
import it.units.malelab.jgea.core.listener.TableBuilder;
import it.units.malelab.jgea.core.solver.Individual;
import it.units.malelab.jgea.core.solver.state.POSetPopulationState;
import it.units.malelab.jgea.core.util.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static it.units.malelab.jgea.core.listener.NamedFunctions.*;

/**
 * @author eric
 */
public class NamedFunctions {

  private static final Logger L = Logger.getLogger(NamedFunctions.class.getName());

  private NamedFunctions() {
  }

  public static List<NamedFunction<? super POSetPopulationState<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>, ?>> basicFunctions() {
    return List.of(iterations(), births(), fitnessEvaluations(), elapsedSeconds());
  }

  public static List<NamedFunction<? super RLEnsembleOutcome, ?>> basicOutcomeFunctions() {
    return List.of(
        f("meanVelocity", "%5.1f", s -> s.results().stream().map(RLEnsembleOutcome.RLOutcome::validationVelocity).reduce(0.0, Double::sum) / s.results().size())
    );
  }

  public static NamedFunction<POSetPopulationState<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>, Individual<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>> best() {
    return ((NamedFunction<POSetPopulationState<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>, Individual<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>>) state -> Misc.first(
        state.getPopulation().firsts())).rename("best");
  }


  public static AccumulatorFactory<POSetPopulationState<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>, BufferedImage, Map<String, Object>> fitnessPlot(
      Function<RLEnsembleOutcome, Double> fitnessFunction
  ) {
    return new TableBuilder<POSetPopulationState<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>, Number, Map<String, Object>>(List.of(
        iterations(),
        f("fitness", fitnessFunction).of(fitness()).of(best()),
        min(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all()),
        median(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all())
    ), List.of()).then(t -> ImagePlotters.xyLines(600, 400).apply(t));
  }

  public static NamedFunction<Pair<POSetPopulationState<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>, Individual<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>>,
      Individual<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>> individualExtractor() {
    return f(
        "individual",
        Pair::second
    );
  }

  public static List<NamedFunction<? super Individual<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>, ?>> individualFunctions(
      Function<RLEnsembleOutcome,
          Double> fitnessFunction
  ) {
    return List.of(
        f("genotype.birth.iteration", "%4d", Individual::genotypeBirthIteration),
        f("fitness", "%5.1f", fitnessFunction).of(fitness())
    );
  }

  public static List<NamedFunction<? super Map<String, Object>, ?>> keysFunctions() {
    return List.of(
        attribute("experiment.name"),
        attribute("seed").reformat("%2d"),
        attribute("terrain"),
        attribute("shape"),
        attribute("sensor.config"),
        attribute("mapper"),
        attribute("transformation"),
        attribute("solver"),
        attribute("episode.time"),
        attribute("episode.transient.time")
    );
  }


  public static List<NamedFunction<? super POSetPopulationState<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>, ?>> populationFunctions(
      Function<RLEnsembleOutcome, Double> fitnessFunction
  ) {
    NamedFunction<? super POSetPopulationState<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>, ?> min = min(Double::compare).of(each(f(
        "fitness",
        fitnessFunction
    ).of(fitness()))).of(all());
    NamedFunction<? super POSetPopulationState<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>, ?> median = median(Double::compare).of(each(f(
        "fitness",
        fitnessFunction
    ).of(fitness()))).of(all());
    return List.of(
    );
  }

  public static Function<POSetPopulationState<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>, Collection<Pair<POSetPopulationState<?, ToDoubleFunction<Grid<Voxel>>,
      RLEnsembleOutcome>, Individual<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>>>> populationSplitter() {
    return state -> {
      List<Pair<POSetPopulationState<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>, Individual<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>>> list = new ArrayList<>();
      state.getPopulation().all().forEach(i -> list.add(Pair.of(state, i)));
      return list;
    };
  }

  public static List<NamedFunction<? super Individual<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>, ?>> serializationFunction(boolean flag) {
    if (!flag) {
      return List.of();
    }
    return List.of(f("serialized", r -> SerializationUtils.serialize(r, SerializationUtils.Mode.GZIPPED_JSON)).of(
        solution()));
  }

  public static NamedFunction<Pair<POSetPopulationState<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>, Individual<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>>,
      POSetPopulationState<?, ToDoubleFunction<Grid<Voxel>>, RLEnsembleOutcome>> stateExtractor() {
    return f(
        "state",
        Pair::first
    );
  }


}
