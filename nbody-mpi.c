#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* Constants */

#define Default_Debug_Acceleration_Scale 100.0f

/* Types */

typedef struct _body {
    float x, y;
    float ax, ay;
    float vx, vy;
    float mass;
} body_t;

/* Globals */

static float debug_acceleration_scale =
    Default_Debug_Acceleration_Scale;

static body_t *bodies = NULL;
static float initial_body_mass;

static float simulation_softening_length_squared;
static float *accelerations = NULL;

/* Utilities */

#ifdef DEBUG

#include <unistd.h>
#include <stdbool.h>

static const char progress_bar[] =
    "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||";
static const int progress_bar_width =
    sizeof(progress_bar) / sizeof(progress_bar[0]);

static bool is_stdout_redirected()
{
    return !isatty(fileno(stdout));
}

static void print_progress_bar(float percentage)
{
    if (!is_stdout_redirected()) { return; }

    int value = (int) (percentage * 100.0f);
    int left_pad = (int) (percentage * progress_bar_width);
    int right_pad = progress_bar_width - left_pad;
    fprintf(
        stderr,
        "\r%3d%% [%.*s%*s]",
        value,
        left_pad,
        progress_bar,
        right_pad,
        ""
    );
    if (value == 100) {
        fprintf(stderr, " \n");
    }
    fflush(stderr);
}

#endif // DEBUG

static float unit_random()
{
    return (float) rand() / (float) RAND_MAX;
}

/* Simulation */

static void generate_debug_data(body_t *bodies, size_t body_count)
{
    for (size_t i = 0; i < body_count; ++i) {
        body_t *body = &bodies[i];

        float angle =
            (float) i / body_count * 2.0f * M_PI +
                (unit_random() - 0.5f) * 0.5f;

        body->x = unit_random();
        body->y = unit_random();
        body->ax = 0.0f;
        body->ay = 0.0f;
        body->mass = initial_body_mass * (unit_random() + 0.5f);

        float random = unit_random();
        body->vx = cosf(angle) * debug_acceleration_scale * random;
        body->vy = sinf(angle) * debug_acceleration_scale * random;
    }
}

static void calculate_newton_gravity_acceleration(
                body_t *first_body, body_t *second_body,
                float *ax, float *ay
            )
{
    *ax = *ay = 0.0f;

    float galactic_plane_r_x =
        second_body->x - first_body->x;
    float galactic_plane_r_y =
        second_body->y - first_body->y;

    float distance_squared =
        (galactic_plane_r_x * galactic_plane_r_x  +
         galactic_plane_r_y * galactic_plane_r_y) +
            simulation_softening_length_squared;
    float distance_squared_cubed =
        distance_squared * distance_squared * distance_squared;
    float inverse =
        1.0f / sqrtf(distance_squared_cubed);
    float scale =
        second_body->mass * inverse;

    *ax += galactic_plane_r_x * scale;
    *ay += galactic_plane_r_y * scale;
}

static void integrate(body_t *body, float delta_time)
{
    body->vx += body->ax * delta_time;
    body->vy += body->ay * delta_time;
    body->x  += body->vx * delta_time;
    body->y  += body->vy * delta_time;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    if (argc < 6) {
        fprintf(
            stderr,
            "Error: incorrect number of arguments\n\n"
            "\tUsage: %s <time period (~10-100)> "
                        "<delta time (~0.01-0.1)> "
                        "<body count (~100-1000)> "
                        "<initial body mass (~10000)> "
                        "<softening length (~100)> "
                        "[debug acceleration scale (~100)]\n",
            argv[0]
        );

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    float time_period = strtof(argv[1], NULL);
    float delta_time  = strtof(argv[2], NULL);
    static const int Base = 10;
    size_t body_count = (size_t) strtol(argv[3], NULL, Base);
    initial_body_mass = strtof(argv[4], NULL);
    float softening_length = strtof(argv[5], NULL);
    simulation_softening_length_squared = softening_length * softening_length;
    if (argc > 6) {
        debug_acceleration_scale = strtof(argv[6], NULL);
    }

    size_t mpi_body_count = body_count + (body_count % world_size ? world_size - body_count % world_size : 0);

    bodies = (body_t *) malloc(sizeof(*bodies) * mpi_body_count);
    assert(bodies != NULL);

    size_t iterations = time_period / delta_time;
    size_t acceleration_entries = iterations * body_count * 2;

    MPI_Datatype mpi_dt_body;
    MPI_Type_contiguous(7, MPI_FLOAT, &mpi_dt_body);
    MPI_Type_commit(&mpi_dt_body);

    if (rank == 0) {
        generate_debug_data(bodies, body_count);

        printf("%zu\n%f\n%f\n", body_count, time_period, delta_time);
        for (size_t i = 0; i < body_count; ++i) {
            body_t *body = &bodies[i];
            printf(
                "%f %f\n"
                "%f %f\n"
                "%f %f\n"
                "%f\n",
                body->x,  body->y,
                body->ax, body->ay,
                body->vx, body->vy,
                body->mass
            );
        }

        accelerations = (float *) malloc(sizeof(*accelerations) * acceleration_entries);
        assert(accelerations != NULL);
    }

    assert(body_count % world_size == 0);
    size_t bodies_per_process = body_count / world_size;
    body_t *sub_bodies = (body_t *) malloc(sizeof(body_t) * bodies_per_process);
    assert(sub_bodies != NULL);


    for (size_t k = 0, next_acceleration = 0; k < iterations; ++k) {
#ifdef DEBUG
        print_progress_bar((float) (k + 1) / iterations);
#endif
        MPI_Bcast(bodies, body_count, mpi_dt_body, 0, MPI_COMM_WORLD);
        size_t begin = rank * bodies_per_process;
        size_t end = begin + bodies_per_process - 1;
        if (end >= body_count) {
            end = body_count - 1;
        }

        for (size_t i = begin, sub_i = 0; i <= end; ++i, ++sub_i) {
            body_t *first_body = &bodies[i];

            float total_ax = 0.0f, total_ay = 0.0f;
            for (size_t j = 0; j < body_count; ++j) {
                body_t *second_body = &bodies[j];
                if (first_body == second_body) {
                    continue;
                }

                float ax, ay;
                calculate_newton_gravity_acceleration(
                    first_body, second_body,
                    &ax, &ay
                );

                total_ax += ax;
                total_ay += ay;
            }

            sub_bodies[sub_i] = *first_body;
            sub_bodies[sub_i].ax = total_ax;
            sub_bodies[sub_i].ay = total_ay;
        }
        
        MPI_Gather(sub_bodies, bodies_per_process, mpi_dt_body, bodies, bodies_per_process, mpi_dt_body, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (size_t i = 0; i < body_count; ++i) {
                body_t *body = &bodies[i];
                integrate(body, delta_time);
                accelerations[next_acceleration++] = body->ax;
                accelerations[next_acceleration++] = body->ay;
            }
        }
    }

    if (rank == 0) {
        for (size_t i = 0; i < acceleration_entries; i += 2) {
            printf("%f %f\n", accelerations[i], accelerations[i + 1]);
        }
    }

    free(accelerations);
    accelerations = NULL;

    free(bodies);
    bodies = NULL;

    free(sub_bodies);
    sub_bodies = NULL;

    MPI_Finalize();
    return EXIT_SUCCESS;
}

