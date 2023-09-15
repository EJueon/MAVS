# -*- coding: utf-8 -*- 
import keras  
import numpy as np
import datetime

from utils.setting import ModelSetting

from art.attacks.evasion import CarliniL2Method, FastGradientMethod, ProjectedGradientDescent # White-box attacks
from art.attacks.evasion import ZooAttack,HopSkipJump # Black-box attacks
from art.estimators.classification import KerasClassifier

testStartTime = datetime.datetime.now() 

class Attacks():
    def __init__(self, dataset: str, metric: str):
        self.dataset = dataset
        self.attack = metric
        self.tile = 1
        self.targeted = False # target_attack, untarget_attack
        self.randInit = False  
        self.nOrigImages = 10
        self.epsilon = 0.2
        self.results = list()
        self.modelSetting = ModelSetting('coverageRate', dataset)
        self.testImage = list()
        self.testLabel = list()
        self.adversarialExamples = None
    
    def generate_samples(self):
        """ adversarial examples 생성 """
        estimator = KerasClassifier(self.modelSetting.model, 
                                    clip_values = (-0.5, 0.5), 
                                    use_logits = True)
        
        if self.attack:
            if self.attack == 'cw':
                attacker = CarliniL2Method(estimator, 
                                           targeted = self.targeted, 
                                           max_iter = 100)
            elif self.attack == 'fgsm_0.2':
                attacker = FastGradientMethod(estimator, 
                                              eps = self.epsilon, 
                                              targeted = self.targeted, 
                                              batch_size = 1)
            elif self.attack == 'pgd_0.2':
                attacker = ProjectedGradientDescent(estimator, 
                                                    eps=self.epsilon, 
                                                    eps_step = self.epsilon * 0.1, 
                                                    targeted = self.targeted, 
                                                    num_random_init = 10)
            elif self.attack == 'zoo':
                attacker = ZooAttack(estimator, 
                                     confidence = 50, 
                                     targeted = self.targeted, 
                                     initial_const = 10000, 
                                     learning_rate = 0.1, 
                                     max_iter = 1000, 
                                     variable_h = 0.5)
            elif self.attack == 'hopskipjump':
                attacker = HopSkipJump(estimator, 
                                       targeted = self.targeted)
            else:
                exit()

        # Tile original images
        for i in range(len(self.modelSetting.image[:self.nOrigImages])):
            for t in range(self.tile):
                if self.targeted:
                    self.testImage.extend([self.modelSetting.image[i] for j in range(9)])
                else:
                    self.testImage.extend([self.modelSetting.image[i]])
        self.testImage = self.modelSetting.preprocess(np.array(self.testImage))
        
        # Tile target labels
        for i in range(len(self.modelSetting.label[:self.nOrigImages])):
            for t in range(self.tile):
                if self.targeted:
                    self.testLabel.extend([j for j in range(10) if j != self.modelSetting.label[i]])
                else:
                    self.testLabel.extend([self.modelSetting.label[i]])
                    
        self.testLabel = keras.utils.to_categorical(self.testLabel, self.modelSetting.numLabels)
        self.adversarialExamples = attacker.generate(self.testImage, self.testLabel)
        
 
    def execute_attack(self):
        """ 공격 및 공격 결과 출력 """
        self.generate_samples()
        predictions = self.modelSetting.model.predict(self.adversarialExamples)
        print(np.max(self.adversarialExamples))
    
        _predict = list()
        for i in range(len(self.adversarialExamples)):
            distortion = np.sum(np.square(self.testImage[i] - self.adversarialExamples[i]))
            index = int(i / (self.tile * (self.numLabels - 1))) if self.targeted else int(i / self.tile)
            if (np.argmax(self.testLabel[i]) == np.argmax(predictions[i]) and self.targeted) or (np.argmax(self.testLabel[i]) != np.argmax(predictions[i]) and not self.targeted):
                print('Original label: %d, Model prediction: %d' % (np.argmax(self.testLabel[i]), np.argmax(predictions[i])))
                print('Distortion: %f' % (distortion))
                
                self.results.append([(self.adversarialExamples[i] + 0.5) * 255, self.modelSetting.label[index], np.argmax(predictions[i])])
                _predict.append(np.argmax(predictions[i]))
            
        print("Acurracy: ", len(_predict)/len(predictions)) 
        

# TEST
# if __name__=="__main__":
#     temp = Attacks('mnist', 'fgsm_0.2') 
#     temp.execute_attack()
        
        
            
            