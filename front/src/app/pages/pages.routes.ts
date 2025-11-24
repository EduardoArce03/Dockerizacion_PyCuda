import { Routes } from '@angular/router';
import { Documentation } from './documentation/documentation';
import { Crud } from './crud/crud';
import { Empty } from './empty/empty';
import { ImageProcessorComponent } from '@/pages/pycuda/component/image-processor.component';

export default [
    { path: 'pycuda', component: ImageProcessorComponent },
    { path: 'documentation', component: Documentation },
    { path: 'crud', component: Crud },
    { path: 'empty', component: Empty },
    { path: '**', redirectTo: '/notfound' },

] as Routes;
