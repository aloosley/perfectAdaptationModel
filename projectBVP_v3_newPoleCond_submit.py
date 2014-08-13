#!/usr/bin/python

import math
import numpy
from numpy import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import *

import scipy.io

from scipy import weave             # for C++
from scipy.weave import converters  # for C++

from progressbar import *           # progress bar

""" Note that these parameters coorespond to my draft report, not the final report."""


class cellSignalling(object):
    # circular cell, units of measure [um, s, nN]
    # Dynamic Quantities: 
        # S(a,s,t) = signal
        # I(r,s,t) = inhibitor
        # A(a,s,t) = activator
        # R(a,s,t) = response
        
    # Parameters    
        # a      = cell radius (um)
        # dr     = radial discretization step (um)
        # ds     = angular discretization step (deg)
        # dt     = time discretization step (s)
        # tf     = end time
        # t_on   = signal one time
        # k_aon  = rate constant
        # k_aoff = rate constant
        # k_ion  = rate constant
        # k_ioff = rate constant
        # k_ron  = rate constant
        # k_roff = rate constant

        
    def __init__(self, a = 10.0, l = 1.0, dr = 0.5, ds = 1.0, dt = 0.01, tf = 12.0, t_on = 1.0, D = 1, k_aon = 1.0, k_aoff = 0.1, k_ion = 1.0, k_ioff = 1.0, k_ron = 1.0, k_roff = 1.0, saveI = 0):
        self.a = a
        self.dr = dr
        self.ds = ds
        self.dt = dt
        self.tf = tf
        
        self.t_on = t_on
        
        self.l = l
        
        self.D = D
        self.k_aon = k_aon
        self.k_aoff = k_aoff
        self.k_ion = k_ion
        self.k_ioff = k_ioff
        self.k_ron = k_ron
        self.k_roff = k_roff

        self.saveI = saveI # binary variable determines if images of I(:,:,t) should be saved

        self.t = 0.0       
        self.tArray = array([])
        self.tArray = append(self.tArray, self.t)
        
        # Making polar meshgrid and initializing dynamic variables
        self.r = arange(0,a+dr,dr) # 0 is center, boundary at r=a
        self.s = arange(0,360,ds) # discretizing by the degree (360 degree circle)
        self.rr, self.ss = meshgrid(self.r,self.s)
        self.S = zeros([int(round(a/dr))+1,int(round(360/ds)),int(round(tf/dt)+1)])
        self.I = zeros([int(round(a/dr))+1,int(round(360/ds)),int(round(tf/dt)+1)])
        #self.I[10:15,0:130,0] = 100
        self.A = self.S.copy()
        self.R = self.S.copy()
        
        # Initial Conditions
        self.S[-1,:,0] = self.drivingSig(self.t) # place signal along membrane
        self.A[-1,:,0] = self.drivingSig(self.t)
        # the IC for self.I is zeros
        
              
    def eulerForwardStep(self):
        
        # set time index (integer)
        t_idx = self.timeIdx(self.t)
        
        # Update driving Signal (optional)
        self.S[-1,:,t_idx+1] = self.drivingSig(self.t+self.dt) # place signal along membrane
               
        # Update A by second order RK (vectors created over all "s" along the membrane at "r=a")
        gA1 = -self.k_aoff*self.A[-1,:,t_idx] + self.k_aon*self.S[-1,:,t_idx]
        gA2 = -self.k_aoff*(self.A[-1,:,t_idx] + (self.dt/2) * gA1) + self.k_aon*self.drivingSig(self.t+self.dt/2)
        gA3 = -self.k_aoff*(self.A[-1,:,t_idx] + (self.dt/2) * gA2) + self.k_aon*self.drivingSig(self.t+self.dt/2)
        gA4 = -self.k_aoff*(self.A[-1,:,t_idx] + self.dt * gA3) + self.k_aon*self.drivingSig(self.t+self.dt)
        self.A[-1,:,t_idx+1] = self.A[-1,:,t_idx] + (self.dt/6) * (gA1 + 2*gA2 + 2*gA3 + gA4)
        
        
        # Update I
        alpha = self.D*self.dt/self.dr**2 
        ds = self.ds
        dr = self.dr
        a = self.a
        I = self.I
        
        iend = int(round(a/dr))
        jend = int(round(360/ds))
        
        # Note here that the first index ranges from 0 to a/dr, the second index ranges from 0 to (360-ds)/ds
        code = """                
                for (int i = 1; i < iend; i++) {
                    for (int j = 0; j < jend; j++) {
                        I(i,j,t_idx+1) = I(i,j,t_idx)*(1-2*alpha-2*alpha/pow(i*ds,2.0)) + alpha*( I(i-1,j,t_idx) + I(i+1,j,t_idx) )*(1+0.5/i) + alpha*( I(i,(j-1)%jend,t_idx) + I(i,(j+1)%jend,t_idx) )*pow(i*ds,-2.0);
                    }
                }
                
                """
        weave.inline(code, ['alpha', 'ds', 'iend', 'jend', 'a', 'I','t_idx'], type_converters=converters.blitz, compiler='gcc')
        
        self.I = I
        # Update I; Neumman BC at center of disk
        # self.I[0,:,t_idx+1] = self.I[1,:,t_idx+1]
        # self.I[0,:,t_idx+1] = mean(self.I[1,:,t_idx+1])
        # self.I[0,:,t_idx+1] = 0.5*(self.I[1,0,t_idx+1] + self.I[1,int(180/self.ds),t_idx+1])
        self.I[0,:,t_idx+1] = 0.5*(self.I[1,0,t_idx] + self.I[1,int(round(180/self.ds)),t_idx])
        
        # self.I[0,:,t_idx+1] = 0
        # Update I; Neumman BC along outside
        tau1 = self.l * self.dr / self.D
        """ This one works, kind of"""
        tau2 = tau1/(1+tau1*self.k_ioff)
        self.I[-1,:,t_idx+1] = self.I[-2, :, t_idx+1] + tau2*self.k_ion*self.S[-1,:,t_idx+1]
                
        """ Neumman BC based on Inhibitor just within the membrane - does not work
        self.I[-1,:,t_idx+1] = self.I[-2,:,t_idx+1] + tau1*(self.k_ion*self.S[-1,:,t_idx+1] - self.k_ioff*self.I[-2,:,t_idx+1])
        """
        
        """ Neumman BC based on Inhibitor using the previous timestep of I... nope
        self.I[-1,:,t_idx+1] = self.I[-2,:,t_idx+1] + tau1*(self.k_ion*self.S[-1,:,t_idx+1] - self.k_ioff*self.I[-1,:,t_idx])
        """
        
        # Update R using Euler (no 2nd order RK because it is hard to calculate g values for I(s))
        self.R[-1,:,t_idx+1] = self.R[-1,:,t_idx] + self.dt*(-self.k_roff*self.I[-1,:,t_idx]*self.R[-1,:,t_idx] + self.k_ron*self.A[-1,:,t_idx]*(1-self.R[-1,:,t_idx]))
        
        
        #Update time
        self.t += self.dt
        self.tArray = append(self.tArray, self.t)
        
        
    def integrate(self):
        
        widgets = ['Integrating:', Percentage(), ' ', Bar(marker='-', left='[', right=']'),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        
        pbar = ProgressBar(widgets=widgets, maxval =int(round(self.tf/self.dt)))
        pbar.start()
        notch = 0
        while self.t < self.tf-self.dt:
            
            self.eulerForwardStep()
            if notch%int(0.05*self.tf/self.dt)==0:
                pbar.update(notch)
                if self.saveI == 1:
                    self.savePolarDiffImage()
#                     self.saveDiffusionImage()
            notch += 1
        
        pbar.finish()
    
    
    
    def saveDiffusionImage(self):
        
        t_idx = self.timeIdx(self.t)
        
        figI = plt.figure(1)
        plotI = plt.imshow(self.I[:,:,t_idx], interpolation='nearest', origin='lower', cmap=cm.jet, aspect='auto', extent=(0,360,0,1))
        plotI_cb = plt.colorbar(plotI)
        plotI_cb.set_clim(0, 0.3)
        plt.title('%0.2f s' % self.t)
        plt.xlabel('s / deg')
        plt.ylabel('r / um')
        
        figName = "Irs%03d.jpg" % t_idx
        figI.savefig(figName)
        close()
        
        
    def savePolarDiffImage(self):
        
        t_idx = self.timeIdx(self.t)
        
        I_xy = zeros([2*int(round(self.a/self.dr))+1,2*int(round(self.a/self.dr))+1])
        for i in range(2*int(round(self.a/self.dr))+1):
            x = i*self.dr - self.a
            for j in range(2*int(round(self.a/self.dr))+1):
                y = j*self.dr - self.a
                
                rho = sqrt(x**2 + y**2)
                angle = 180.0 * arctan2(y,x)/math.pi + 180.0
                
                rho_idx = round(rho/self.dr)
                angle_idx = round(angle/self.ds)
                
                if rho_idx < self.a/self.dr+1 and angle_idx < 360/self.ds:
                    I_xy[2*int(round(self.a/self.dr))-i,j] = self.I[rho_idx,angle_idx,t_idx]
                    
                
        figI = plt.figure(1)
        plotI = plt.imshow(I_xy[:,:], origin='lower', cmap=cm.jet, aspect='auto', extent=(-self.a,self.a,-self.a,self.a))
        plotI_cb = plt.colorbar(plotI)
        plotI_cb.set_clim(0, 0.3)
        plt.title('%0.2f s' % self.t)
        plt.xlabel('x / um')
        plt.ylabel('y / um')
        
        figName = "Ixy%03d.jpg" % t_idx
        figI.savefig(figName)
        close()
                
        
    # Signal and time index methods        
    def drivingSig(self,t):
        
        if t < self.t_on:
            sig = zeros([len(arange(0,360,self.ds))])
        elif t >= self.t_on:
            sig = cos(2*pi*arange(0,360,self.ds)/720)**2 # COSINE SIGNAL
            
        return sig   
        
        
    def timeIdx(self,t):
        # set time index (integer)
        t_idx = int(round(t / self.dt))
        return t_idx
    
    
    """ Output Variables and Plotting Methods """
    def outputDynamics(self):
        arrayDiff = len(self.tArray) - len(self.S[-1,0,:])
        if arrayDiff >= 1:
            self.tArrayPlot = self.tArray[:-arrayDiff]
        elif arrayDiff <= -1:
            print 'tArray error!!! arrayDiff = '+str(arrayDiff)
        elif arrayDiff == 0:
            self.tArrayPlot = self.tArray.copy()
        
        self.frontS = self.S[-1,0,:]
        self.frontA = self.A[-1,0,:]
        self.frontR = self.R[-1,0,:]
        self.frontI = self.I[-1,0,:]
        
        self.backS = self.S[-1,int(round(180/self.ds)),:]
        self.backA = self.A[-1,int(round(180/self.ds)),:]
        self.backR = self.R[-1,int(round(180/self.ds)),:]
        self.backI = self.I[-1,int(round(180/self.ds)),:]

        self.totS = self.frontS.copy()
        self.totA = self.frontA.copy()
        self.totR = self.frontR.copy()
        self.totI = self.frontI.copy()
        
        for k in xrange(0,int(round(self.tf/self.dt)),1):
            self.totS[k] = sum(self.S[-1,:,k])
            self.totA[k] = sum(self.A[-1,:,k])
            self.totR[k] = sum(self.R[-1,:,k])
            self.totI[k] = sum(self.I[-1,:,k])
    
    
    def inhibitorPlot(self):
        
        """
        figI1 = plt.figure()
        plotI = plt.imshow(self.I[:,:,-1], interpolation='bilinear', origin='lower',
                cmap=cm.jet)
        plt.plotI_cb = plt.colorbar(plotI, orientation='horizonal',shrink = 0.8)
        # plt.plotI_cb.set_clim(0, 1)
        plt.title('2D Inhibitor plot (end time)')
        plt.ylabel('r (um)')
        plt.xlabel('theta (deg)')
        """
                
        """
        figI3 = plt.figure()
        plotIt = plt.imshow(self.I[-1,:,:], interpolation='bilinear', origin='lower', cmap=cm.jet)
        plotIt_cb = plt.colorbar(plotIt)
        """
        
    def timePlots(self):
        anglePlot = 25
        
        figTP1 = plt.figure(figsize=(5, 3))
        plt.plot(self.tArrayPlot,self.S[-1,int(round(anglePlot/self.ds)),:],'--')
        plt.plot(self.tArrayPlot,self.A[-1,int(round(anglePlot/self.ds)),:],'k-')
        plt.plot(self.tArrayPlot,self.R[-1,int(round(anglePlot/self.ds)),:],'r-')
        plt.plot(self.tArrayPlot,self.I[-1,int(round(anglePlot/self.ds)),:],'b-')
        plt.legend(['Source','Activator','Response','Inhibitor'],loc=2)
        plt.axis([0,self.tf,0,4])
        #plt.title('Response at the front of the cell')
        #plt.xlabel('time / AU')
        #plt.ylabel('concentration / AU')
        
        savefig('sigDynamicsFront.eps', format='eps')

        """
        figTP2 = plt.figure()
        plt.plot(self.tArrayPlot,self.totS,'--')
        plt.plot(self.tArrayPlot,self.totA,'--')
        plt.plot(self.tArrayPlot,self.totR,'--')
        plt.plot(self.tArrayPlot,self.totI,'--')
        plt.legend(['Source','Activator','Response','Inhibitor'],loc=2)
        plt.axis([0,self.tf,0,4])
        plt.title('Integrated Concentrations (at membrane)')
        plt.xlabel('time / AU')
        plt.ylabel('concentration / AU')
        """
        
        figTP3 = plt.figure(figsize=(4, 5))
        axTPA = figTP3.add_subplot(3,1,1)
        axTPA.set_yticks(arange(0,361,60))
        #plt.title('Activator')
        plotTPA = axTPA.imshow(self.A[-1,:,:], interpolation='bilinear', origin='lower', cmap=cm.jet, aspect='auto', extent=(0,self.tf,0,360))
        plotTPA_cb = plt.colorbar(plotTPA, ticks=arange(0,1.1,0.5))
        plotTPA_cb.set_clim(0, 1)
        axTPA.plot([0, self.tf],[anglePlot, anglePlot],'w--')
        axTPA.axis([0,self.tf,0,360])
        #ylabel('s / deg')
        
        axTPI = figTP3.add_subplot(3,1,2)
        axTPI.set_yticks(arange(0,361,60))
        #plt.title('Inhibitor')
        plotTPI = axTPI.imshow(self.I[-1,:,:], interpolation='bilinear', origin='lower', cmap=cm.jet, aspect='auto', extent=(0,self.tf,0,360))
        plotTPI_cb = plt.colorbar(plotTPI, ticks=arange(0, 121, 40))
        plotTPI_cb.set_clim(0, 1)
        axTPI.plot([0, self.tf],[anglePlot, anglePlot],'w--')
        axTPI.axis([0,self.tf,0,360])
        #ylabel('s / deg')
        
        
        axTPR = figTP3.add_subplot(3,1,3)
        axTPR.set_yticks(arange(0,361,60))
        #plt.title('Response')
        plotTPR = axTPR.imshow(self.R[-1,:,:], interpolation='bilinear', origin='lower', cmap=cm.jet, aspect='auto', extent=(0,self.tf,0,360))
        plotTPR_cb = plt.colorbar(plotTPR, ticks = arange(0,round(100.0*self.R[-1,:,:].max())/100.0,round(100.0*self.R[-1,:,:].max()/5.0)/100.0))
        plotTPR_cb.set_clim(0, 1)
        axTPR.plot([0, self.tf],[anglePlot, anglePlot],'w--')
        axTPR.axis([0,self.tf,0,360])
        #xlabel('time / AU')
        #ylabel('s / deg')
        
        savefig('sigDynamics.eps', format='eps')

    def rosePlots(self):
        figRP1 = plt.figure()
        axRP1 = figRP1.add_subplot(111, polar=True)
        
        axRP1.bar(self.s*math.pi/180,self.S[-1,:,-1], width = 0.01)

# Begin model
t_on = 0.5
tf = 13.0
a = 10.0
dt, dr, ds = 0.01, 0.5, 1.0
l, D = 1.0, 1.0
k_aon, k_aoff, k_ion, k_ioff, k_ron, k_roff = 1.0, 1.0, 1.0, 10.0, 1.0, 1.0

cell = cellSignalling(a = a, l = l, dr = dr, ds = ds, dt = dt, tf = tf, t_on = t_on, D = D,
                      k_aon=k_aon,k_aoff=k_aoff,k_ion=k_ion,k_ioff=k_ioff,k_ron=k_ron,k_roff=k_roff,
                      saveI=1)
cell.integrate()
cell.outputDynamics()
cell.timePlots()

plt.show()
