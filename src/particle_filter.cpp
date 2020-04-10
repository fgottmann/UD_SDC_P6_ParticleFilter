/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

// src stackoverflow
// https://stackoverflow.com/questions/10847007/using-the-gaussian-probability-density-function-in-c
double normal_2d_pdf(double x0, double x1, double s[2])
{
  constexpr static const double inv_sqrt_2pi = 0.3989422804014327;
  double a0 = x0 / s[0];
  double a1 = x1 / s[1];

  return inv_sqrt_2pi / (s[0]*s[1]) * std::exp(-0.5 *( a0*a0 + a1*a1));
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  // cleanup first
  particles.clear();

  num_particles = 500;  //  Set the number of particles

  std::default_random_engine generator;
  std::normal_distribution<double> distribution_x(x,std[0]);
  std::normal_distribution<double> distribution_y(y,std[1]);
  std::normal_distribution<double> distribution_theta(theta,std[2]);

  for (int ii = 0; ii < num_particles; ii++)
  {
    Particle p;
    p.id 		= ii;
    p.x  		= distribution_x(generator);
    p.y  		= distribution_y(generator);
    p.theta  	= distribution_theta(generator);
    p.weight  = 1;
    particles.push_back(p);
  }

  // initialization done
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::default_random_engine generator;
  std::normal_distribution<double> distribution_x(0,std_pos[0]);
  std::normal_distribution<double> distribution_y(0,std_pos[1]);
  std::normal_distribution<double> distribution_theta(0,std_pos[2]);

  for (unsigned int ii_p = 0; ii_p < particles.size(); ii_p++)
  {
    double theta_eff = particles[ii_p].theta;
    particles[ii_p].theta += yaw_rate*delta_t + distribution_theta(generator);

    // use the average angle during integration
    theta_eff = 0.5*(theta_eff + particles[ii_p].theta);
    particles[ii_p].x     += velocity*cos(theta_eff)*delta_t + distribution_x(generator);
    particles[ii_p].y     += velocity*sin(theta_eff)*delta_t + distribution_y(generator);

    //std::cout << particles[ii_p].x << " " << particles[ii_p].y << " "<< particles[ii_p].theta << std::endl;
  }

}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  vector<LandmarkObs> predicted;
  // iterate through all map nodes to find relevant ones
  for (unsigned int ii_m = 0; ii_m < map_landmarks.landmark_list.size(); ii_m++)
    for (unsigned int ii_p = 0; ii_p < particles.size(); ii_p++)
    {
      double dx = fabs(map_landmarks.landmark_list[ii_m].x_f - particles[ii_p].x);
      if (dx <= sensor_range)
      {
        double dy = fabs(map_landmarks.landmark_list[ii_m].y_f - particles[ii_p].y);
        if (dy <= sensor_range)
        {
          if (dx*dx + dy*dy <= sensor_range*sensor_range) // avoiding sqrt in range_check
          {
            LandmarkObs obs;
            obs.id = map_landmarks.landmark_list[ii_m].id_i;
            obs.x = map_landmarks.landmark_list[ii_m].x_f;
            obs.y = map_landmarks.landmark_list[ii_m].y_f;
            predicted.push_back(obs);
            break; // ok found one -> next
          }
        }
      }
    }


  // after found relevant landmark's send them to association
  for (unsigned int ii_p = 0; ii_p < particles.size(); ii_p++)
  {
    double c_theta = cos(particles[ii_p].theta);
    double s_theta = sin(particles[ii_p].theta);
    particles[ii_p].weight = 0;

    particles[ii_p].sense_x.clear();
    particles[ii_p].sense_y.clear();
    particles[ii_p].associations.clear();

    for (unsigned int ii_o = 0; ii_o < observations.size(); ii_o++)
    {
      // transform pose to global frame
      double g_x = particles[ii_p].x + c_theta*observations[ii_o].x
          - s_theta*observations[ii_o].y;
      double g_y = particles[ii_p].y + s_theta*observations[ii_o].x
          + c_theta*observations[ii_o].y;

      // initialize association and current node weight
      int association = -1;
      double cur_weight = 0;

      // iterate trough valid map data
      for (unsigned int ii_m = 0; ii_m < predicted.size(); ii_m++)
      {
        double dx = g_x-predicted[ii_m].x;
        double dy = g_y-predicted[ii_m].y;
        if (fabs(dx) < std_landmark[0]*3 && fabs(dy) < std_landmark[1]*3) // set trust region to avoid unneccessary calculations (3-sigma-range)
        {
          double prob = normal_2d_pdf(dx,dy,std_landmark);
          if (prob > cur_weight)
          {
            cur_weight = prob;
            association = predicted[ii_m].id;
          }
          if (cur_weight > 0.95) // quality i.o.? then stop here.
            break;
        }
      }

      // if we found an association -> write data
      if (association >= 0)
      {
        //std::cout << cur_weight << "test" <<association<< std::endl;
        particles[ii_p].weight += cur_weight;
        particles[ii_p].sense_x.push_back(g_x);
        particles[ii_p].sense_y.push_back(g_y);
        particles[ii_p].associations.push_back(association); // -1 indicates not associated
      }
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // copy weights into vector
  weights.clear();
  for (unsigned int ii_p = 0; ii_p < particles.size(); ii_p++)
  {
    //std::cout << ii_p << " test "<< particles[ii_p].weight << std::endl;
    weights.push_back(particles[ii_p].weight);
  }

  // initialize random variable and weighted, random distribution
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(weights.begin(),weights.end());

  // append most probable particles
  unsigned nb_particles = particles.size();
  for (unsigned int ii_p = 0; ii_p < nb_particles; ii_p++)
  {
    particles.push_back(particles[d(gen)]);
  }

  // remove original particles
  particles.erase (particles.begin(),particles.begin()+nb_particles);


}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
