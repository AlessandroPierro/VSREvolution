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

import it.units.erallab.hmsrobots.tasks.rllocomotion.RLEnsembleOutcome;
import it.units.erallab.hmsrobots.util.SerializableFunction;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.malelab.jgea.core.listener.AccumulatorFactory;
import it.units.malelab.jgea.core.listener.NamedFunction;
import it.units.malelab.jgea.core.listener.TableBuilder;
import it.units.malelab.jgea.core.solver.Individual;
import it.units.malelab.jgea.core.solver.state.POSetPopulationState;
import it.units.malelab.jgea.core.util.ImagePlotters;
import it.units.malelab.jgea.core.util.Misc;
import it.units.malelab.jgea.core.util.Pair;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.logging.Logger;

import static it.units.malelab.jgea.core.listener.NamedFunctions.*;

/**
 * @author eric
 */
public class NamedFunctions {

    private static final Logger L = Logger.getLogger(NamedFunctions.class.getName());

    private NamedFunctions() {
    }

    public static List<NamedFunction<? super POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, ?>> basicFunctions() {
        return List.of(iterations(), births(), fitnessEvaluations(), elapsedSeconds());
    }

    public static List<NamedFunction<? super RLEnsembleOutcome, ?>> basicOutcomeFunctions() {
        return List.of(
                f("meanVelocity", "%5.1f", s -> s.results().stream().map(RLEnsembleOutcome.RLOutcome::validationVelocity).reduce(0.0, Double::sum) / s.results().size())
        );
    }

    public static NamedFunction<POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, Individual<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>> best() {
        return ((NamedFunction<POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, Individual<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>>) state -> Misc.first(
                state.getPopulation().firsts())).rename("best");
    }


    public static AccumulatorFactory<POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, BufferedImage, Map<String, Object>> fitnessPlot(
            Function<RLEnsembleOutcome, Double> fitnessFunction
    ) {
        return new TableBuilder<POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, Number, Map<String, Object>>(List.of(
                iterations(),
                f("fitness", fitnessFunction).of(fitness()).of(best()),
                min(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all()),
                median(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all())
        ), List.of()).then(t -> ImagePlotters.xyLines(600, 400).apply(t));
    }

    public static NamedFunction<Pair<POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, Individual<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>>,
            Individual<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>> individualExtractor() {
        return f(
                "individual",
                Pair::second
        );
    }

    public static List<NamedFunction<? super Individual<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, ?>> individualFunctions(
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


    public static List<NamedFunction<? super POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, ?>> populationFunctions(
            Function<RLEnsembleOutcome, Double> fitnessFunction
    ) {
        NamedFunction<? super POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, ?> min = min(Double::compare).of(each(f(
                "fitness",
                fitnessFunction
        ).of(fitness()))).of(all());
        NamedFunction<? super POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, ?> median = median(Double::compare).of(each(f(
                "fitness",
                fitnessFunction
        ).of(fitness()))).of(all());
        return List.of(
        );
    }

    public static Function<POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, Collection<Pair<POSetPopulationState<?, SerializableFunction<double[], Double>,
            RLEnsembleOutcome>, Individual<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>>>> populationSplitter() {
        return state -> {
            List<Pair<POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, Individual<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>>> list = new ArrayList<>();
            state.getPopulation().all().forEach(i -> list.add(Pair.of(state, i)));
            return list;
        };
    }

    public static List<NamedFunction<? super Individual<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, ?>> serializationFunction(boolean flag) {
        if (!flag) {
            return List.of();
        }
        return List.of(f("serialized", r -> SerializationUtils.serialize(r, SerializationUtils.Mode.GZIPPED_JSON)).of(
                solution()));
    }

    public static NamedFunction<Pair<POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, Individual<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>>,
            POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>> stateExtractor() {
        return f(
                "state",
                Pair::first
        );
    }


}
