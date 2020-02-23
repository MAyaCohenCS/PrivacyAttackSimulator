import attacks_lib
import private_mechanisms_lib
import data_utils

import matplotlib.pyplot as plt
import numpy as np

ATTACKS = [attacks_lib.LP_reconstructor]

BASELINES = [private_mechanisms_lib.No_privacy,
              private_mechanisms_lib.Random_Answers]

MECHANISMS = [private_mechanisms_lib.Round_to_R_multiplication,
              private_mechanisms_lib.Epsilon_gausian_noise]


COMPARATORS = [data_utils.census_citizenship_DB_DC, data_utils.intro_grades_DB_DC]

QUERY_CONST_FACTOR = 2
REPETITIONS = 50
EPSILON_SAMPLE_STEP_SIZE = 1


def draw_graphics(dc_class):

    graphics_file_name = 'simulation_results/graphics_data/' + str(dc_class.__name__) + ".csv"
    graphics_file = open(graphics_file_name, "r")

    epsilon_range =[int(s) for s in graphics_file.readline()[:-1].split(',')]
    baselines = graphics_file.readline()[:-1].split(',')
    results = graphics_file.readline()[:-1].split(',')

    fig = plt.figure()
    ax = plt.subplot(211)

    for data_series in baselines:
        ax.plot(epsilon_range, [float(graphics_file.readline()[:-1])]*len(epsilon_range), label=data_series.strip())

    for data_series in results:
        ax.plot(epsilon_range, [float(s) for s in graphics_file.readline()[:-1].split(',')], '.-', label=data_series.strip())

    ax.set(xlabel='Security parameter', ylabel='Reconstruction rate',
           title= dc_class().get_data_name()+' Reconstruction rates by security parameter')
    ax.grid()
    from matplotlib.font_manager import FontProperties
    #
    fontP = FontProperties()
    fontP.set_size('small')
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0 * 0.7, chartBox.width, chartBox.height * 1.5])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), shadow=True)

    ax.axis(ymax= 1.03, ymin=-0.03)
    fig.savefig("simulation_results/graphics_data/"+str(dc_class.__name__)+" results.png")
    # plt.show()

def simulate (dc_class, a_class, pm_class, epsilon_range, DC, query_limit, graphics_file):

    # log simulations
    log_file = open('simulation_results/' + str(dc_class.__name__) + '/' + str(a_class.__name__) + "_vs_" + str(
        pm_class.__name__) + ".csv", "w")
    log_file.write("epsilon: ,repetitions\n")

    # collects data for graphics
    results = np.zeros((len(epsilon_range), REPETITIONS))

    for e in range(len(epsilon_range)):
        log_file.write(str(epsilon_range[e]) + ": ")

        for r in range(REPETITIONS):
            A = a_class(DC.size, DC.get_data_bounds())
            PM = pm_class(epsilon_range[e], DC.data, DC.get_data_bounds())

            for _ in range(query_limit):
                A.learn_from_response(PM.respond_query(A.generate_query()))

            attack_success_rate = DC.score(A.predict_origin())
            log_file.write(',' + str(attack_success_rate))
            results[e, r] = attack_success_rate

        log_file.write('\n')
        log_file.flush()
    log_file.close()

    graphics_file.write(str(np.median(results, axis=1).tolist()).strip('[]') + '\n')
    graphics_file.flush()

def build_simulation():

    # set a separate environment for each DB
    for dc_class in COMPARATORS:

        DC = dc_class()
        epsilon_range = np.arange(1,DC.size,EPSILON_SAMPLE_STEP_SIZE)
        query_limit = int(QUERY_CONST_FACTOR * DC.size)

        #collect graphics data
        graphics_file_name = 'simulation_results/graphics_data/' + str(dc_class.__name__) + ".csv"
        graphics_file = open(graphics_file_name, "w")
        graphics_file.write(str(epsilon_range.tolist()).strip('[]')+'\n')
        graphics_file.flush()

        legend = [str(A_class.__name__) + " vs. " + str(PM_class.__name__) for A_class in ATTACKS for PM_class in BASELINES]
        graphics_file.write(', '.join(legend)+ '\n')
        graphics_file.flush()

        legend = [str(A_class.__name__) + " vs. " + str(PM_class.__name__) for A_class in ATTACKS for PM_class in MECHANISMS]
        graphics_file.write(', '.join(legend)+ '\n')
        graphics_file.flush()

        #pick a pair <A, PM> for the simulation
        for a_class in ATTACKS:

            for bln_class in BASELINES:
                simulate(dc_class, a_class, bln_class, [0], DC, query_limit, graphics_file)

            for pm_class in MECHANISMS:
                simulate(dc_class, a_class, pm_class, epsilon_range, DC, query_limit, graphics_file)

        graphics_file.close()


def main():
    #build_simulation()

    for dc_class in COMPARATORS:
        draw_graphics(dc_class)


if __name__ == "__main__":
    main()




