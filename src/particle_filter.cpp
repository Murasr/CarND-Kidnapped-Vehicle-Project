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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  //! Assume the number of particles
  num_particles = 50; 
  
  std::default_random_engine rand_engine;

  // Creating Normal distributions for x,y and theta
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  //! Using normal distribution initialize all the particles
  for(int32_t count = 0; count < num_particles; count++)
  {
    Particle particle;
    particle.id = count;
    particle.x = dist_x(rand_engine);
    particle.y = dist_y(rand_engine);
    particle.theta = dist_theta(rand_engine);
    particle.weight = 1;
    
    particles.push_back(particle);
  }
  
  is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {

  
  std::default_random_engine rand_engine;

  // Creating Normal distributions for x,y and theta to add noise
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);
  
  for(int32_t count = 0; count < particles.size(); count++)
  {
    //! Store the last position values
    double old_x = particles[count].x;
    double old_y = particles[count].y;
    double old_theta = particles[count].theta;
    
    //! yaw rate not equal to zero
    if(fabs(yaw_rate) > 0.000001)
    {
        double yawrate_delta = yaw_rate * delta_t;
    	double new_x = old_x + velocity/yaw_rate * (sin(old_theta + yawrate_delta) - sin(old_theta)) ;
        double new_y = old_y + velocity/yaw_rate * (cos(old_theta) - cos(old_theta + yawrate_delta)) ;
        double new_theta = old_theta + yawrate_delta;

        //! Assign the predicted position with added noise
        particles[count].x = new_x + dist_x(rand_engine);
        particles[count].y = new_y + dist_y(rand_engine);
        particles[count].theta = new_theta + dist_theta(rand_engine);
    }
    else //! yaw rate equal to zero
    {
        //! update x and y based on right angle triangle  formulas
        double new_x = old_x + velocity * delta_t * cos(old_theta);
        double new_y = old_y + velocity * delta_t * sin(old_theta);
        double new_theta = old_theta;
      
        //! Assign the predicted position with added noise
        particles[count].x = new_x + dist_x(rand_engine);
        particles[count].y = new_y + dist_y(rand_engine);
        particles[count].theta = new_theta + dist_theta(rand_engine);
    }
  }
}

void ParticleFilter::dataAssociation(const Map &map_landmarks, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for(int32_t count = 0; count < observations.size(); count++)
  {
    //! From all map landmarks, find the nearest neighbour landmark to the observation and assign the id to it
    double minDistance = std::numeric_limits<double>::max();
    for(int32_t landCount = 0; landCount < map_landmarks.landmark_list.size(); landCount++)
    {
      double distance = dist(observations[count].x, observations[count].y, map_landmarks.landmark_list[landCount].x_f, map_landmarks.landmark_list[landCount].y_f);
      if(distance < minDistance)
      {
        minDistance = distance;
        observations[count].id = map_landmarks.landmark_list[landCount].id_i;
        
        //std::cout<< "data ass : " << observations[count].id << std::endl;
      }
    }
  }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {   
  //! Clear the old weights
  m_weights.clear();
  
  //! For each particle
  for(int32_t count = 0; count < particles.size(); count++)
  {
    vector<LandmarkObs> conv_observations;
    
    double veh_pos_x = particles[count].x;
    double veh_pos_y = particles[count].y;
    double veh_theta = particles[count].theta;
    
    //! Convert the observations from vehicle coordinates to map coordinates
    for(int32_t obs_count = 0; obs_count < observations.size(); obs_count++)
    {
      double obs_pos_x = observations[obs_count].x;
      double obs_pos_y = observations[obs_count].y;
      
      LandmarkObs conv_obs;
      conv_obs.x = veh_pos_x + (cos(veh_theta) * obs_pos_x) - (sin(veh_theta) * obs_pos_y);
      conv_obs.y = veh_pos_y + (sin(veh_theta) * obs_pos_x) + (cos(veh_theta) * obs_pos_y);
      
      conv_observations.push_back(conv_obs);
    }
    
    //! do data association
    dataAssociation(map_landmarks, conv_observations);
    
    //! Clear the old values
    particles[count].associations.clear();
    particles[count].sense_x.clear();
    particles[count].sense_y.clear();
    
    //! Assign the new observation values
    for(int32_t obs_count = 0; obs_count < conv_observations.size(); obs_count++)
    {
      particles[count].associations.push_back(conv_observations[obs_count].id);
      particles[count].sense_x.push_back(conv_observations[obs_count].x);
      particles[count].sense_y.push_back(conv_observations[obs_count].y);
    }
    
    //! update weights
    double weight = 1;
    double prefix_mul = 1.0/(2 * M_PI * std_landmark[0] * std_landmark[1]);
    //! For each associated observation, do the following
    for(int32_t ass_count = 0; ass_count < particles[count].associations.size(); ass_count++)
    {
      double mx = particles[count].sense_x[ass_count];
      double my = particles[count].sense_y[ass_count];
      
      //! find the mx and my from map landmarks based on associated id
      for(int32_t landCount = 0; landCount < map_landmarks.landmark_list.size(); landCount++)
      {
        if(particles[count].associations[ass_count] == map_landmarks.landmark_list[landCount].id_i)
        {
          mx = map_landmarks.landmark_list[landCount].x_f;
          my = map_landmarks.landmark_list[landCount].y_f;
          break;
        }
      }
      
      double x = particles[count].sense_x[ass_count];
      double y = particles[count].sense_y[ass_count];
      
      double expo = exp(-1.0 * ( ((x-mx)*(x-mx)/(2 * std_landmark[0]* std_landmark[0])) + ((y-my)*(y-my)/(2 * std_landmark[1]* std_landmark[1])) ));
      weight *= (prefix_mul) * expo;
    }
       
    //! Store the weight into the particle
    particles[count].weight = weight;      
    
    //! Store the weight into a separate list
    m_weights.push_back(weight);
  }
}

void ParticleFilter::resample() {
  //! Resample is done based on discrete distribution method which made my life easy 
  std::default_random_engine rand_engine;
  std::discrete_distribution<int32_t> disc_dist(m_weights.begin(), m_weights.end());
  
 // static int print_once = 0;
  
  std::vector<Particle> resampled_particles;
  for(int32_t count = 0; count < particles.size(); count++)
  {
    int32_t index = disc_dist(rand_engine);
    resampled_particles.push_back(particles[index]);
  //  if(print_once < 5)
  //  {
   // 	std::cout << "count: " << count << " index: " << index << " weight: " << m_weights[index] << std::endl;
   // }
  }
  
 // print_once++;
  
  particles = resampled_particles;

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